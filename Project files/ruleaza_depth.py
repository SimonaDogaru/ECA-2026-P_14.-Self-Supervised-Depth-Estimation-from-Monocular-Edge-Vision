#!/usr/bin/env python3
"""
FAZELE 9-10 pe placă: inferență student pe NPU + demo (hartă de adâncime colorată)
+ raport de resurse (cât din placă ocupă modelul: timp, CPU, NPU, memorie, stocare,
temperatură, putere).

Scriptul e independent (doar numpy, Pillow, tflite_runtime; OpenCV opțional).
Copiază pe placă: acest fișier + modelul .tflite + contractul lui .json (același
nume, generat de 05_export.py) (+ imagini / fișierele ref_*.npy).

Optimizări față de versiunea inițială (ruleaza_depth_original.py), rezultat identic:
  * imaginea rămâne uint8 de la citire până la NPU (fără conversie la float32);
  * cuantizarea intrării (+ normalizarea RGB la modelele *_rgb) = un tabel de 256
    valori calculat o singură dată; pentru scala 1 / zero-point -128 devine un XOR;
  * decuantizarea ieșirii + conversia în metri = un alt tabel de 256 valori;
  * detaliile tensorilor se citesc o singură dată, nu la fiecare cadru;
  * redimensionarea se face doar dacă imaginea nu are deja rezoluția modelului;
  * --paralel: citirea + preprocesarea cadrului următor rulează pe alt fir cât
    timp NPU-ul lucrează la cadrul curent.

Exemple (pe placă, prin SSH):
  # 1. verificare: placa dă aceeași ieșire ca PC-ul? (+ calea rapidă = calea veche)
  python3 ruleaza_depth.py --model student_ghidat_int8.tflite --npu \
          --verifica ref_input.npy ref_output_ghidat_int8.npy
  # 2. măsurare de performanță + raport de resurse (fără salvarea imaginilor)
  python3 ruleaza_depth.py --model student_ghidat_int8.tflite --npu --intrare demo_imagini \
          --fara_salvare --bucle 5
  # 3. același lucru cu citirea/preprocesarea în paralel cu NPU-ul
  python3 ruleaza_depth.py --model student_ghidat_int8.tflite --npu --intrare demo_imagini \
          --fara_salvare --bucle 5 --paralel
  # 4. demo pe imagini salvate (rezultate în ./rezultate)
  python3 ruleaza_depth.py --model student_ghidat_int8.tflite --npu --intrare demo_imagini
  # 5. comparație CPU (fără --npu)
  python3 ruleaza_depth.py --model student_ghidat_int8.tflite --intrare demo_imagini --fara_salvare
  # 6. cameră (USB /dev/video0) cu afișare pe HDMI, dacă OpenCV are suport de afișare
  python3 ruleaza_depth.py --model student_ghidat_int8.tflite --npu --intrare camera:0 --afiseaza
  # Putere: placa nu are de obicei senzori de curent; cu un wattmetru USB-C pe alimentare:
  python3 ruleaza_depth.py ... --putere_w 4.1 --putere_repaus_w 2.6
"""
import argparse
import glob
import json
import os
import queue
import re
import shutil
import sys
import threading
import time

import numpy as np
from PIL import Image

# ---- Contractul modelului. Valorile implicite = cod/config.py; dacă lângă model
# există <model>.json (scris de 05_export.py), valorile se citesc de acolo.
# Rezoluția se citește oricum din model.
CONTRACT = {"q_min": 0.006, "q_max": 0.3, "max_depth": 50.0,
            # Camera KITTI (2011_09_26): fx [px], baseline [m], lățime originală [px]
            "fx": 721.5377, "baseline": 0.5327, "orig_w": 1242}
IMG_H, IMG_W = 192, 640
DELEGATE = "/usr/lib/libvx_delegate.so"


def load_contract(model):
    p = os.path.splitext(model)[0] + ".json"
    if os.path.exists(p):
        with open(p) as f:
            CONTRACT.update(json.load(f))
        print(f"Contract citit din {p}")
    else:
        print(f"ATENȚIE: lipsește {p}; folosesc valorile implicite {CONTRACT}")


def make_interpreter(model, use_npu):
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:  # pe PC
        from ai_edge_litert.interpreter import Interpreter, load_delegate
    delegates = []
    if use_npu:
        os.environ["USE_GPU_INFERENCE"] = "0"   # NPU, nu GPU (documentație NXP)
        delegates = [load_delegate(DELEGATE)]
    it = Interpreter(model_path=model, experimental_delegates=delegates, num_threads=4)
    it.allocate_tensors()
    return it


# =============================================================================
# Calea de referință (cea veche, pe float32). Folosită doar la --verifica, ca să
# demonstrăm că tabelele dau exact același rezultat.
# =============================================================================
def normalize_rgb(x):
    """Doar pentru modelele *_int8_rgb: fiecare canal R/G/B pe propriul interval (ca tflite_runner.preprocess)."""
    if "rgb_lo" not in CONTRACT:
        return x
    lo = np.asarray(CONTRACT["rgb_lo"], np.float32)
    hi = np.asarray(CONTRACT["rgb_hi"], np.float32)
    return np.clip((x - lo) * (255.0 / (hi - lo)), 0.0, 255.0)


def quantize_input(x, det):
    if np.issubdtype(det["dtype"], np.integer):
        s, z = det["quantization"]
        info = np.iinfo(det["dtype"])
        return np.clip(np.round(x / s) + z, info.min, info.max).astype(det["dtype"])
    return x.astype(det["dtype"])


def dequantize_output(y, det):
    if np.issubdtype(det["dtype"], np.integer):
        s, z = det["quantization"]
        return s * (y.astype(np.float32) - z)
    return y.astype(np.float32)


def s_to_depth(s):
    lo, hi = np.log(CONTRACT["q_min"]), np.log(CONTRACT["q_max"])
    q = np.exp(lo + s * (hi - lo))
    return CONTRACT["fx"] * CONTRACT["baseline"] / (q * CONTRACT["orig_w"])


# =============================================================================
# Calea rapidă: tot ce se poate calcula o dată se calculează în constructor.
# =============================================================================
class DepthModel:
    def __init__(self, it):
        self.it = it
        self.inp = it.get_input_details()[0]
        self.out = it.get_output_details()[0]
        self.h, self.w = (int(v) for v in self.inp["shape"][1:3])
        self._build_input_lut()
        self._build_output_lut()

    def _build_input_lut(self):
        """Pixel uint8 (0..255) -> valoarea cuantizată pe care o așteaptă modelul."""
        codes = np.arange(256, dtype=np.float32)
        if "rgb_lo" in CONTRACT:            # normalizare diferită pe fiecare canal
            lut = quantize_input(normalize_rgb(np.repeat(codes[:, None], 3, 1)), self.inp)  # [256, 3]
        else:
            lut = quantize_input(normalize_rgb(codes), self.inp)                            # [256]
        self.in_lut = lut
        self.in_mode = "lut"
        if lut.ndim == 1 and lut.dtype == np.int8 and np.array_equal(
                lut, (np.arange(256, dtype=np.uint8) ^ 0x80).view(np.int8)):
            self.in_mode = "xor"            # scala 1, zero-point -128: int8 = uint8 XOR 0x80
        elif lut.ndim == 1 and lut.dtype == np.uint8 and np.array_equal(
                lut, np.arange(256, dtype=np.uint8)):
            self.in_mode = "identic"        # modelul primește direct uint8
        self._ch = np.arange(3)

    def _build_output_lut(self):
        """Cod de ieșire (int8/uint8) -> adâncime în metri, într-un singur tabel."""
        dt = np.dtype(self.out["dtype"])
        self.out_lut = None
        if dt.itemsize == 1 and np.issubdtype(dt, np.integer):
            codes = np.arange(256, dtype=np.uint8).view(dt)        # ordinea vederii uint8
            self.out_lut = s_to_depth(dequantize_output(codes, self.out)).astype(np.float32)

    def quantize(self, x_u8):
        """x_u8: [H, W, 3] uint8 RGB -> tensor de intrare [1, H, W, 3]."""
        if self.in_mode == "xor":
            q = np.bitwise_xor(x_u8, 0x80).view(np.int8)
        elif self.in_mode == "identic":
            q = x_u8
        elif self.in_lut.ndim == 2:
            q = self.in_lut[x_u8, self._ch]
        else:
            q = self.in_lut[x_u8]
        return q[None]

    def run(self, q):
        """Doar NPU/CPU: set_tensor + invoke + get_tensor. Întoarce ieșirea brută [h, w]."""
        self.it.set_tensor(self.inp["index"], q)
        self.it.invoke()
        return self.it.get_tensor(self.out["index"])[0, ..., 0]

    def to_depth(self, y):
        if self.out_lut is not None:
            return self.out_lut[y.view(np.uint8)]
        return s_to_depth(dequantize_output(y, self.out))

    def to_s(self, y):
        return dequantize_output(y, self.out)


def preprocess(img, h, w):
    """
    La fel ca la antrenare: RGB, redimensionare biliniară la w x h, valori [0, 255] - dar
    rămânem pe uint8. img = imagine PIL sau cadru OpenCV (numpy BGR).
    """
    if isinstance(img, np.ndarray):                # cadru de cameră (BGR)
        import cv2
        if img.shape[:2] != (h, w):
            down = img.shape[0] > h or img.shape[1] > w
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if down else cv2.INTER_LINEAR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# =============================================================================
# Vizualizare (nu intră în timpii măsurați)
# =============================================================================
def depth_to_unit(depth):
    """Adâncime [m] -> [0, 1] pe scară log (1 = ~1 m, 0 = max_depth), aceeași pentru predicție și GT."""
    near = CONTRACT["fx"] * CONTRACT["baseline"] / (CONTRACT["q_max"] * CONTRACT["orig_w"])
    d = np.clip(depth, near, CONTRACT["max_depth"])
    return np.log(CONTRACT["max_depth"] / d) / np.log(CONTRACT["max_depth"] / near)


def turbo(x):
    """Paleta 'turbo' (aproximare polinomială Google), x în [0,1] -> RGB uint8."""
    x = np.clip(x, 0, 1)[..., None]
    c = np.concatenate([
        0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943)))),
        0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604)))),
        0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973)))),
    ], -1)
    return (np.clip(c, 0, 1) * 255).astype(np.uint8)


def visualize(x_u8, depth_full, gt=None):
    """Sus imaginea de intrare, apoi harta prezisă, apoi (dacă există) ground truth-ul LiDAR KITTI."""
    panels = [x_u8, turbo(depth_to_unit(depth_full))]
    if gt is not None:
        g = turbo(depth_to_unit(np.where(gt > 0, gt, CONTRACT["max_depth"])))
        g[gt <= 0] = 0                       # negru = fără punct LiDAR
        panels.append(g)
    return np.concatenate(panels, 0)


def load_gt(path):
    """<imagine>_gt.png (06_imagini_demo.py): PNG 16 biți, adâncime = valoare / 256 (convenția KITTI)."""
    if not path or not os.path.exists(path):
        return None
    gt = np.asarray(Image.open(path), dtype=np.float32) / 256.0
    if gt.shape != (IMG_H, IMG_W):
        return None
    return gt


def upsample(depth, h, w):
    return np.asarray(Image.fromarray(depth.astype(np.float32)).resize((w, h), Image.BILINEAR))


# =============================================================================
# Surse de cadre. Fiecare element: (nume, imagine încărcată, cale GT, timp citire [s])
# =============================================================================
def frames_from(source, loops=1):
    if source.startswith("camera"):
        import cv2
        idx = int(source.split(":")[1]) if ":" in source else 0
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            sys.exit(f"Nu pot deschide camera {idx}")
        while True:
            a = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            yield f"cam_{int(time.time() * 1000)}", frame, None, time.perf_counter() - a
        return
    paths = [source] if os.path.isfile(source) else sorted(
        glob.glob(os.path.join(source, "*.png")) + glob.glob(os.path.join(source, "*.jpg")))
    paths = [p for p in paths if not os.path.splitext(p)[0].endswith("_gt")]
    if not paths:
        sys.exit(f"Nu am găsit imagini în {source}")
    for _ in range(max(1, loops)):
        for p in paths:
            base = os.path.splitext(p)[0]
            a = time.perf_counter()
            img = Image.open(p)
            img.load()                           # decodarea PNG/JPG se întâmplă aici
            yield os.path.basename(base), img, base + "_gt.png", time.perf_counter() - a


# =============================================================================
# Citirea resurselor plăcii (toate "best effort": dacă un fișier lipsește, se sare)
# =============================================================================
def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def proc_status():
    out = {}
    txt = _read("/proc/self/status") or ""
    for key in ("VmHWM", "VmRSS"):
        m = re.search(rf"^{key}:\s+(\d+)", txt, re.M)
        if m:
            out[key] = int(m.group(1)) / 1024
    return out


def meminfo():
    out = {}
    txt = _read("/proc/meminfo") or ""
    for key in ("MemTotal", "MemAvailable", "CmaTotal", "CmaFree"):
        m = re.search(rf"^{key}:\s+(\d+)", txt, re.M)
        if m:
            out[key] = int(m.group(1)) / 1024
    return out


def cpu_stat():
    """Linia 'cpu' din /proc/stat -> (timp ocupat, timp total) în jiffies, pentru toată placa."""
    txt = _read("/proc/stat")
    if not txt:
        return None
    v = [int(x) for x in txt.splitlines()[0].split()[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return sum(v) - idle, sum(v)


def temperatures():
    out = {}
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        t, name = _read(z + "/temp"), _read(z + "/type")
        if t and t.strip().lstrip("-").isdigit():
            out[(name or os.path.basename(z)).strip()] = int(t) / 1000
    return out


def cpu_freqs_mhz():
    cur, mx = [], []
    for c in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq")):
        a, b = _read(c + "/scaling_cur_freq"), _read(c + "/cpuinfo_max_freq")
        if a and b:
            cur.append(int(a) / 1000)
            mx.append(int(b) / 1000)
    return cur, mx


def hwmon_power_w():
    """Senzori de putere/curent expuși de kernel (de ex. INA2xx). Pe multe plăci nu există."""
    total, found = 0.0, []
    for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = (_read(d + "/name") or os.path.basename(d)).strip()
        for p in sorted(glob.glob(d + "/power*_input")):
            v = _read(p)
            if v and v.strip().isdigit():
                w = int(v) / 1e6                  # µW -> W
                total += w
                found.append(f"{name}:{os.path.basename(p)}={w:.2f} W")
    return (total, found) if found else (None, [])


def npu_idle_counters():
    """
    Contoarele driverului galcore (NPU-ul VeriSilicon de pe i.MX8M Plus):
    /sys/kernel/debug/gc/idle dă timpii On/Off/Idle/Suspend de la ultima citire.
    Necesită root și debugfs montat; altfel întoarce None.
    """
    txt = _read("/sys/kernel/debug/gc/idle")
    if not txt:
        return None
    vals = {k.lower(): int(v) for k, v in re.findall(r"(\w+)\s*:\s*(\d+)", txt)}
    return vals if "on" in vals else None


class PowerSampler(threading.Thread):
    """Eșantionează senzorii hwmon de putere la ~5 Hz, dacă există."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self._halt = [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            w, _ = hwmon_power_w()
            if w is not None:
                self.samples.append(w)
            self._halt.wait(0.2)

    def stop(self):
        self._halt.set()


# =============================================================================
# Raport
# =============================================================================
def candidate_resolutions(h, w):
    """Rezoluții cu laturi multiplu de 32, raport de aspect ca acum (KITTI ~3.3:1)."""
    aspect = w / h
    out = []
    for hh in range(128, 513, 32):
        ww = int(round(hh * aspect / 32)) * 32
        out.append((hh, ww))
    return out


def print_report(r, args):
    ms = lambda v: float(np.mean(v)) * 1000 if len(v) else 0.0  # noqa: E731
    t = r["timpi_ms"]
    budget = 1000.0 / args.fps_tinta
    frame_ms = r["cadru_ms"]
    fps = 1000.0 / frame_ms
    line = "-" * 66

    print(f"\n{'=' * 66}\n RAPORT  {r['model']}  |  {r['dispozitiv']}  |  {r['mod']}\n"
          f" {r['cadre']} cadre măsurate (după încălzire), rezoluție {r['rezolutie']}\n{'=' * 66}")

    print(f"\n[1] TIMP pe cadru{line[16:]}")
    print(f"  citire/decodare imagine   {t['citire']:7.2f} ms   (cu camera: captura, în paralel cu restul)")
    print(f"  preprocesare (uint8)      {t['preprocesare']:7.2f} ms")
    print(f"  {'cuantizare (' + r['cuantizare'] + ')':<26}{t['cuantizare']:7.2f} ms")
    print(f"  NPU/CPU (invoke + I/O)    {t['inferenta']:7.2f} ms   (mediană {t['inferenta_mediana']:.2f}, "
          f"p95 {t['inferenta_p95']:.2f})")
    print(f"  postprocesare (tabel)     {t['postprocesare']:7.2f} ms")
    if r["mod"] == "paralel":
        print(f"  => măsurat pe ceas (fir citire+preprocesare în paralel): {frame_ms:.2f} ms/cadru"
              f"  ->  {fps:.1f} FPS")
    else:
        print(f"  => total secvențial fără citire: {frame_ms:.2f} ms  ->  {fps:.1f} FPS"
              f"   |  cu citire: {frame_ms + t['citire']:.2f} ms -> {1000 / (frame_ms + t['citire']):.1f} FPS")
    share = t["inferenta"] / frame_ms * 100
    print(f"  din timpul pe cadru, {share:.0f}% e NPU/CPU (rețeaua), {100 - share:.0f}% e cod Python/numpy")

    print(f"\n[2] BUGET la {args.fps_tinta:g} FPS ({budget:.1f} ms/cadru){line[32:]}")
    used = frame_ms / budget * 100
    print(f"  ocupat: {used:.0f}% din buget  |  rezervă: {budget / frame_ms:.2f}x  ->  "
          + ("ÎNCAPE" if used <= 100 else "NU încape"))
    if r["npu_activ_pct"] is not None:
        print(f"  NPU activ (contor galcore): {r['npu_activ_pct']:.0f}% din timp")
    else:
        print(f"  NPU ocupat ≈ {t['inferenta'] * fps / 10:.0f}% din timp la {fps:.1f} FPS "
              f"(estimat din timpul de invoke)")
    print(f"  dacă rulezi exact la {args.fps_tinta:g} FPS, NPU-ul e ocupat ≈ "
          f"{min(100, t['inferenta'] / budget * 100):.0f}% și stă liber restul")

    print(f"\n[3] CÂT MAI ÎNCAPE (estimare liniară cu nr. de pixeli; confirmă cu benchmark.sh){line[78:]}")
    px0 = r["h"] * r["w"]
    print(f"  {'rezoluție':>10} {'pixeli':>7} {'NPU est.':>9} {'cadru est.':>11} {'FPS est.':>9}  la {args.fps_tinta:g} FPS")
    for hh, ww in candidate_resolutions(r["h"], r["w"]):
        k = hh * ww / px0
        if k < 0.5 or k > 4.5:
            continue
        f_ms = frame_ms * k
        tag = "acum" if (hh, ww) == (r["h"], r["w"]) else (
            "ok" if f_ms <= 0.8 * budget else ("la limită" if f_ms <= budget else "nu"))
        print(f"  {hh:>4}x{ww:<5} {k:>6.2f}x {t['inferenta'] * k:>7.1f}ms {f_ms:>9.1f}ms {1000 / f_ms:>9.1f}  {tag}")
    print("  ('ok' = sub 80% din buget; la rezoluții mari NPU-ul poate scala mai prost decât liniar)")

    print(f"\n[4] CPU{line[7:]}")
    print(f"  procesul folosește {r['cpu_proces_nuclee']:.2f} din {r['nuclee']} nuclee "
          f"({r['cpu_proces_nuclee'] / r['nuclee'] * 100:.0f}% din CPU-ul plăcii)")
    if r["cpu_placa_pct"] is not None:
        print(f"  încărcare totală placă în timpul rulării: {r['cpu_placa_pct']:.0f}%")
    if r["frecventa_mhz"]:
        cur, mx = r["frecventa_mhz"]
        print(f"  frecvență CPU: {min(cur):.0f}-{max(cur):.0f} MHz (max {max(mx):.0f} MHz)"
              + ("  <- posibilă limitare termică/energetică" if max(cur) < 0.9 * max(mx) else ""))

    print(f"\n[5] MEMORIE{line[11:]}")
    m = r["memorie_mb"]
    if "VmHWM" in m and "MemTotal" in m:
        print(f"  proces: maxim {m['VmHWM']:.0f} MB (acum {m.get('VmRSS', 0):.0f} MB) din "
              f"{m['MemTotal']:.0f} MB RAM = {m['VmHWM'] / m['MemTotal'] * 100:.1f}%")
        print(f"  RAM disponibil pe placă: {m.get('MemAvailable', float('nan')):.0f} MB")
    if m.get("cma_npu") is not None:
        print(f"  memorie CMA ocupată de NPU (buffere driver): {m['cma_npu']:.1f} MB din "
              f"{m['CmaTotal']:.0f} MB CMA")

    print(f"\n[6] STOCARE{line[11:]}")
    s = r["stocare"]
    print(f"  model .tflite: {s['model_kb']:.0f} kB  |  contract .json: {s['json_kb']:.1f} kB  |  "
          f"script: {s['script_kb']:.0f} kB")
    print(f"  total necesar pe placă: {s['total_kb']:.0f} kB = {s['total_kb'] / 1024 / s['disc_total_gb'] / 1024 * 100:.4f}% "
          f"din {s['disc_total_gb']:.1f} GB (liber: {s['disc_liber_gb']:.1f} GB)")

    print(f"\n[7] TEMPERATURĂ & PUTERE{line[24:]}")
    for k, (a, b) in r["temperatura_c"].items():
        print(f"  {k}: {a:.1f} °C -> {b:.1f} °C ({b - a:+.1f})")
    if not r["temperatura_c"]:
        print("  (nu există senzori în /sys/class/thermal)")
    p = r["putere"]
    if p["sursa"] == "hwmon":
        print(f"  putere medie (senzori hwmon): {p['w']:.2f} W  ->  {p['mj_cadru']:.1f} mJ/cadru, "
              f"{p['fps_pe_w']:.1f} FPS/W")
    elif p["sursa"] == "wattmetru":
        print(f"  putere în rulare (wattmetru): {p['w']:.2f} W  ->  {p['mj_cadru']:.1f} mJ/cadru, "
              f"{p['fps_pe_w']:.1f} FPS/W")
        if p.get("dinamic_w") is not None:
            print(f"  peste repaus: +{p['dinamic_w']:.2f} W  ->  {p['mj_cadru_dinamic']:.1f} mJ/cadru "
                  f"atribuiți modelului")
    else:
        print("  putere: placa nu expune senzori de curent (hwmon). Măsoară cu un wattmetru USB-C pe")
        print("  alimentare (în repaus și în timpul rulării) și dă --putere_w W --putere_repaus_w W.")
    print()


def _p(v, q):
    return float(np.percentile(v, q)) * 1000 if len(v) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npu", action="store_true", help="rulează cu delegate-ul VX (NPU)")
    ap.add_argument("--intrare", default=None, help="imagine, folder sau camera:0")
    ap.add_argument("--iesire", default="rezultate")
    ap.add_argument("--afiseaza", action="store_true", help="fereastră OpenCV (HDMI)")
    ap.add_argument("--max", type=int, default=0, help="număr maxim de cadre (0 = toate)")
    ap.add_argument("--bucle", type=int, default=1, help="repetă setul de imagini de N ori (măsurare mai stabilă)")
    ap.add_argument("--fara_salvare", action="store_true", help="nu salva imaginile (măsurare curată)")
    ap.add_argument("--paralel", action="store_true", help="citire+preprocesare pe alt fir, în paralel cu NPU")
    ap.add_argument("--fps_tinta", type=float, default=30.0, help="FPS-ul dorit (pentru bugetul de timp)")
    ap.add_argument("--putere_w", type=float, default=None, help="putere măsurată cu wattmetru în timpul rulării [W]")
    ap.add_argument("--putere_repaus_w", type=float, default=None, help="putere în repaus [W]")
    ap.add_argument("--verifica", nargs=2, metavar=("REF_IN", "REF_OUT"))
    args = ap.parse_args()

    global IMG_H, IMG_W
    load_contract(args.model)
    mem0 = meminfo()
    t0 = time.perf_counter()
    it = make_interpreter(args.model, args.npu)
    t_init = time.perf_counter() - t0
    model = DepthModel(it)
    IMG_H, IMG_W = model.h, model.w
    dev = "NPU (VX delegate)" if args.npu else "CPU (4 fire)"
    print(f"Model: {args.model} ({IMG_H}x{IMG_W}) | rulare pe: {dev} | "
          f"inițializare: {t_init * 1000:.0f} ms | cuantizare intrare: {model.in_mode}")

    if args.verifica:
        x = np.load(args.verifica[0])
        ref = np.load(args.verifica[1])[0, ..., 0]
        a = time.perf_counter()
        y = model.run(quantize_input(normalize_rgb(x.astype(np.float32)), model.inp))
        print(f"Primul apel (include compilarea grafului pe NPU): {(time.perf_counter() - a) * 1000:.0f} ms")
        s = model.to_s(y)
        d = np.abs(s - ref)
        print(f"Diferență față de PC: medie {d.mean():.4f}, max {d.max():.4f} "
              f"(un pas de cuantizare ~ {1 / 256:.4f})")
        print("OK" if d.mean() < 0.01 else "ATENȚIE: ieșirea diferă de PC")
        # Calea rapidă trebuie să dea exact aceleași numere ca cea veche.
        if x.dtype == np.uint8 or np.array_equal(x, np.round(x)):
            x_u8 = np.clip(x, 0, 255).astype(np.uint8)[0]
            q_old = quantize_input(normalize_rgb(x_u8.astype(np.float32)), model.inp)
            q_new = model.quantize(x_u8)[0]
            y_new = model.run(model.quantize(x_u8))
            d_old, d_new = s_to_depth(model.to_s(y_new)), model.to_depth(y_new)
            ok_q = np.array_equal(q_old, q_new)
            ok_d = np.allclose(d_old, d_new, rtol=1e-5, atol=1e-5)
            print(f"Cuantizare rapidă ({model.in_mode}) = cea veche: {'DA' if ok_q else 'NU'} | "
                  f"tabel adâncime = formula veche: {'DA' if ok_d else 'NU'}")
        return

    if not args.intrare:
        sys.exit("Dă --intrare sau --verifica")
    if not args.fara_salvare and not args.afiseaza:
        os.makedirs(args.iesire, exist_ok=True)

    # ---- Încălzire: primul cadru include compilarea grafului pe NPU.
    warm = model.quantize(np.zeros((IMG_H, IMG_W, 3), np.uint8))
    a = time.perf_counter()
    model.run(warm)
    t_warm = time.perf_counter() - a
    print(f"Primul invoke (compilare graf NPU, exclus din medii): {t_warm * 1000:.0f} ms")
    model.run(warm)
    mem1 = meminfo()

    t_read, t_pre, t_q, t_inf, t_post, gt_errs = [], [], [], [], [], []
    temp0, cpu0, times0 = temperatures(), cpu_stat(), os.times()
    npu_idle_counters()                                   # resetează contorul galcore
    sampler = PowerSampler()
    if hwmon_power_w()[0] is not None:
        sampler.start()

    def produce(src):
        """citire -> preprocesare -> cuantizare. Rulează pe firul principal sau pe un fir separat."""
        for name, img, gt_path, tr in src:
            a = time.perf_counter()
            x = preprocess(img, IMG_H, IMG_W)
            b = time.perf_counter()
            q = model.quantize(x)
            c = time.perf_counter()
            yield name, x, q, gt_path, tr, b - a, c - b

    source = frames_from(args.intrare, args.bucle)
    if args.paralel:
        qbuf = queue.Queue(maxsize=2)

        def worker():
            for item in produce(source):
                qbuf.put(item)
            qbuf.put(None)

        threading.Thread(target=worker, daemon=True).start()
        items = iter(qbuf.get, None)
    else:
        items = produce(source)

    n = 0
    loop_start = time.perf_counter()
    for name, x, q, gt_path, tr, tp, tq in items:
        a = time.perf_counter()
        y = model.run(q)
        b = time.perf_counter()
        depth = model.to_depth(y)
        c = time.perf_counter()
        t_read.append(tr)
        t_pre.append(tp)
        t_q.append(tq)
        t_inf.append(b - a)
        t_post.append(c - b)
        n += 1

        if not args.fara_salvare or args.afiseaza:
            # Vizualizarea (și GT-ul) NU intră în timpii pe etape: sunt doar pentru demo.
            depth_full = upsample(depth, IMG_H, IMG_W)
            gt = load_gt(gt_path)
            vis = visualize(x, depth_full, gt)
            err = ""
            if gt is not None:
                m = (gt > 0) & (gt < CONTRACT["max_depth"])
                if m.any():
                    p = np.clip(depth_full[m], 1e-3, CONTRACT["max_depth"])
                    gt_errs.append(np.mean(np.abs(p - gt[m]) / gt[m]))
                    err = f" | AbsRel vs LiDAR {gt_errs[-1]:.3f}"
            h, w = depth.shape
            print(f"{name}: NPU {(b - a) * 1000:.1f} ms | adâncime centru ~ {depth[h // 2, w // 2]:.1f} m{err}")
            k = max(1, 960 // IMG_W)
            if args.afiseaza:
                import cv2
                recent = slice(-10, None)
                fr = (np.mean(t_pre[recent]) + np.mean(t_q[recent]) + np.mean(t_inf[recent])
                      + np.mean(t_post[recent]))
                big = cv2.resize(vis[..., ::-1], (vis.shape[1] * k, vis.shape[0] * k),
                                 interpolation=cv2.INTER_NEAREST)
                cv2.putText(big, f"{dev}  {1 / fr:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                cv2.imshow("Adancime monoculara - student", big)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                Image.fromarray(vis).resize((vis.shape[1] * k, vis.shape[0] * k), Image.NEAREST).save(
                    os.path.join(args.iesire, f"{name}_depth.png"))
        elif n % 50 == 0:
            print(f"  {n} cadre...")
        if args.max and n >= args.max:
            break
    wall = time.perf_counter() - loop_start

    sampler.stop()
    times1, cpu1, temp1 = os.times(), cpu_stat(), temperatures()
    gc_idle = npu_idle_counters()
    if not t_inf:
        sys.exit("Niciun cadru procesat.")

    ms = lambda v: float(np.mean(v)) * 1000  # noqa: E731
    seq_ms = ms(t_pre) + ms(t_q) + ms(t_inf) + ms(t_post)
    frame_ms = (wall / n * 1000) if args.paralel else seq_ms
    cpu_s = (times1.user - times0.user) + (times1.system - times0.system)

    npu_pct = None
    if gc_idle:
        tot = sum(gc_idle.get(k, 0) for k in ("on", "off", "idle", "suspend"))
        npu_pct = gc_idle["on"] / tot * 100 if tot else None

    mem = {**proc_status(), **mem1}
    if "CmaFree" in mem0 and "CmaFree" in mem1:
        mem["cma_npu"] = mem0["CmaFree"] - mem1["CmaFree"]

    du = shutil.disk_usage(os.path.dirname(os.path.abspath(args.model)))
    json_path = os.path.splitext(args.model)[0] + ".json"
    sto = {"model_kb": os.path.getsize(args.model) / 1024,
           "json_kb": os.path.getsize(json_path) / 1024 if os.path.exists(json_path) else 0.0,
           "script_kb": os.path.getsize(os.path.abspath(__file__)) / 1024,
           "disc_total_gb": du.total / 1024 ** 3, "disc_liber_gb": du.free / 1024 ** 3}
    sto["total_kb"] = sto["model_kb"] + sto["json_kb"] + sto["script_kb"]

    fps = 1000.0 / frame_ms
    power = {"sursa": None}
    if sampler.samples:
        w = float(np.mean(sampler.samples))
        power = {"sursa": "hwmon", "w": w}
    elif args.putere_w:
        power = {"sursa": "wattmetru", "w": args.putere_w}
    if power["sursa"]:
        power["mj_cadru"] = power["w"] / fps * 1000
        power["fps_pe_w"] = fps / power["w"]
        if args.putere_repaus_w is not None and power["sursa"] == "wattmetru":
            power["dinamic_w"] = power["w"] - args.putere_repaus_w
            power["mj_cadru_dinamic"] = power["dinamic_w"] / fps * 1000

    freqs = cpu_freqs_mhz()
    report = {
        "model": os.path.basename(args.model), "dispozitiv": dev,
        "mod": "paralel" if args.paralel else "secvential",
        "cadre": n, "rezolutie": f"{IMG_H}x{IMG_W}", "h": IMG_H, "w": IMG_W,
        "init_ms": t_init * 1000, "primul_invoke_ms": t_warm * 1000,
        "cuantizare": model.in_mode,
        "timpi_ms": {"citire": ms(t_read), "preprocesare": ms(t_pre), "cuantizare": ms(t_q),
                     "inferenta": ms(t_inf), "inferenta_mediana": float(np.median(t_inf)) * 1000,
                     "inferenta_p95": _p(t_inf, 95), "postprocesare": ms(t_post)},
        "cadru_ms": frame_ms, "fps": fps, "fps_tinta": args.fps_tinta,
        "buget_ocupat_pct": frame_ms / (1000.0 / args.fps_tinta) * 100,
        "npu_activ_pct": npu_pct,
        "nuclee": os.cpu_count() or 1, "cpu_proces_nuclee": cpu_s / wall,
        "cpu_placa_pct": ((cpu1[0] - cpu0[0]) / (cpu1[1] - cpu0[1]) * 100)
        if cpu0 and cpu1 and cpu1[1] > cpu0[1] else None,
        "frecventa_mhz": freqs if freqs[0] else None,
        "memorie_mb": mem, "stocare": sto,
        "temperatura_c": {k: (temp0[k], temp1[k]) for k in temp0 if k in temp1},
        "putere": power,
        "absrel_lidar": float(np.mean(gt_errs)) if gt_errs else None,
    }
    print_report(report, args)
    if gt_errs:
        print(f"AbsRel mediu față de LiDAR (0-{CONTRACT['max_depth']:.0f} m, la rezoluția modelului, "
              f"{len(gt_errs)} imagini): {np.mean(gt_errs):.3f}")
    elif args.fara_salvare:
        print("(AbsRel nu se calculează cu --fara_salvare; rulează fără el pentru calitate.)")

    os.makedirs(args.iesire, exist_ok=True)
    tag = "npu" if args.npu else "cpu"
    out_json = os.path.join(args.iesire, f"raport_{os.path.splitext(report['model'])[0]}_{tag}"
                                         f"{'_paralel' if args.paralel else ''}.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    print(f"Raport salvat: {out_json}")


if __name__ == "__main__":
    main()
