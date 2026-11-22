#!/usr/bin/env python3
"""
FAZELE 9-10 pe placă: inferență student pe NPU + demo (hartă de adâncime colorată).

Scriptul e independent (doar numpy, Pillow, tflite_runtime; OpenCV opțional).
Copiază pe placă: acest fișier + modelul .tflite + contractul lui .json (același
nume, generat de 05_export.py) (+ imagini / fișierele ref_*.npy).

Ieșirea demo: imaginea de intrare, harta prezisă și, dacă lângă imagine există
<nume>_gt.png (generat de 06_imagini_demo.py), ground truth-ul LiDAR KITTI.
Harta și GT-ul folosesc aceeași scară de culori (roșu = aproape, albastru = ~50 m,
negru = fără măsurătoare LiDAR), deci se pot compara direct.

Exemple (pe placă, prin SSH):
  # 1. verificare: placa dă aceeași ieșire ca PC-ul?
  python3 ruleaza_depth.py --model student_ghidat_int8.tflite --npu \
          --verifica ref_input.npy ref_output_ghidat_int8.npy
  # 2. demo pe imagini salvate (rezultate în ./rezultate)
  python3 ruleaza_depth.py --model student_ghidat_int8.tflite --npu --intrare demo_imagini
  # 3. comparație CPU (fără --npu)
  python3 ruleaza_depth.py --model student_ghidat_int8.tflite --intrare demo_imagini
  # 4. cameră (USB /dev/video0) cu afișare pe HDMI, dacă OpenCV are suport de afișare
  python3 ruleaza_depth.py --model student_ghidat_int8.tflite --npu --intrare camera:0 --afiseaza
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

import json

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


def preprocess(img_pil):
    """La fel ca la antrenare: RGB, redimensionare biliniară la IMG_W x IMG_H, valori [0, 255]."""
    img = img_pil.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None]


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


def infer(it, x):
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], quantize_input(normalize_rgb(x), i))
    it.invoke()
    return dequantize_output(it.get_tensor(o["index"]), o)[0, ..., 0]   # s, [48,160]


def s_to_depth(s):
    lo, hi = np.log(CONTRACT["q_min"]), np.log(CONTRACT["q_max"])
    q = np.exp(lo + s * (hi - lo))
    return CONTRACT["fx"] * CONTRACT["baseline"] / (q * CONTRACT["orig_w"])


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


def visualize(x, depth, gt=None):
    """
    Sus imaginea de intrare, apoi harta prezisă, apoi (dacă există) ground truth-ul
    LiDAR KITTI. Scară fixă în metri (roșu = aproape, albastru închis = ~50 m),
    nu pe percentile, ca predicția și GT-ul să fie comparabile.
    """
    color = turbo(depth_to_unit(depth))
    panels = [x[0].astype(np.uint8),
              np.asarray(Image.fromarray(color).resize((IMG_W, IMG_H), Image.NEAREST))]
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


def peak_memory_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def frames_from(source):
    if source.startswith("camera"):
        import cv2
        idx = int(source.split(":")[1]) if ":" in source else 0
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            sys.exit(f"Nu pot deschide camera {idx}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"cam_{int(time.time() * 1000)}", Image.fromarray(frame[..., ::-1]), None
    else:
        paths = [source] if os.path.isfile(source) else sorted(
            glob.glob(os.path.join(source, "*.png")) + glob.glob(os.path.join(source, "*.jpg")))
        for p in paths:
            base = os.path.splitext(p)[0]
            if base.endswith("_gt"):
                continue                       # fișierele ground truth nu sunt intrări
            yield os.path.basename(base), Image.open(p), base + "_gt.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npu", action="store_true", help="rulează cu delegate-ul VX (NPU)")
    ap.add_argument("--intrare", default=None, help="imagine, folder sau camera:0")
    ap.add_argument("--iesire", default="rezultate")
    ap.add_argument("--afiseaza", action="store_true", help="fereastră OpenCV (HDMI)")
    ap.add_argument("--max", type=int, default=0, help="număr maxim de cadre (0 = toate)")
    ap.add_argument("--verifica", nargs=2, metavar=("REF_IN", "REF_OUT"))
    args = ap.parse_args()

    global IMG_H, IMG_W
    load_contract(args.model)
    t0 = time.perf_counter()
    it = make_interpreter(args.model, args.npu)
    t_init = time.perf_counter() - t0
    IMG_H, IMG_W = (int(v) for v in it.get_input_details()[0]["shape"][1:3])
    dev = "NPU (VX delegate)" if args.npu else "CPU (4 fire)"
    print(f"Model: {args.model} ({IMG_H}x{IMG_W}) | rulare pe: {dev} | "
          f"inițializare: {t_init * 1000:.0f} ms")

    if args.verifica:
        x = np.load(args.verifica[0]).astype(np.float32)
        ref = np.load(args.verifica[1])[0, ..., 0]
        s = infer(it, x)          # primul apel: include compilarea grafului pe NPU
        d = np.abs(s - ref)
        print(f"Diferență față de PC: medie {d.mean():.4f}, max {d.max():.4f} "
              f"(un pas de cuantizare ~ {1 / 256:.4f})")
        print("OK" if d.mean() < 0.01 else "ATENȚIE: ieșirea diferă de PC")
        return

    if not args.intrare:
        sys.exit("Dă --intrare sau --verifica")
    os.makedirs(args.iesire, exist_ok=True)
    t_pre, t_inf, t_post = [], [], []
    gt_errs = []
    for n, (name, img, gt_path) in enumerate(frames_from(args.intrare)):
        a = time.perf_counter()
        x = preprocess(img)
        b = time.perf_counter()
        s = infer(it, x)
        c = time.perf_counter()
        depth = s_to_depth(s)
        d = time.perf_counter()
        # Vizualizarea (și GT-ul) NU intră în timpul măsurat: sunt doar pentru demo.
        depth_full = np.asarray(Image.fromarray(depth.astype(np.float32)).resize(
            (IMG_W, IMG_H), Image.BILINEAR))
        gt = load_gt(gt_path)
        vis = visualize(x, depth_full, gt)
        err = ""
        if gt is not None:
            m = (gt > 0) & (gt < CONTRACT["max_depth"])
            if m.any():
                p = np.clip(depth_full[m], 1e-3, CONTRACT["max_depth"])
                gt_errs.append(np.mean(np.abs(p - gt[m]) / gt[m]))
                err = f" | AbsRel vs LiDAR {gt_errs[-1]:.3f}"
        if n == 0:
            print(f"Primul cadru (încălzire NPU, exclus din medie): {(c - b) * 1000:.1f} ms")
        else:
            t_pre.append(b - a)
            t_inf.append(c - b)
            t_post.append(d - c)
        h, w = depth.shape
        print(f"{name}: inferență {(c - b) * 1000:.1f} ms | adâncime centru ~ "
              f"{depth[h // 2, w // 2]:.1f} m{err}")
        if args.afiseaza:
            import cv2
            fps = 1.0 / max(np.mean(t_pre[-10:] + t_inf[-10:] + t_post[-10:]) * 3, 1e-6) if t_inf else 0
            k = max(1, 960 // IMG_W)
            big = cv2.resize(vis[..., ::-1], (vis.shape[1] * k, vis.shape[0] * k),
                             interpolation=cv2.INTER_NEAREST)
            cv2.putText(big, f"{dev}  {fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Adancime monoculara - student", big)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            k = max(1, 960 // IMG_W)
            Image.fromarray(vis).resize((vis.shape[1] * k, vis.shape[0] * k), Image.NEAREST).save(
                os.path.join(args.iesire, f"{name}_depth.png"))
        if args.max and n + 1 >= args.max:
            break

    if t_inf:
        ms = lambda v: np.mean(v) * 1000  # noqa: E731
        total = ms(t_pre) + ms(t_inf) + ms(t_post)
        print("\n=== Rezumat ({}, {} cadre după încălzire) ===".format(dev, len(t_inf)))
        print(f"preprocesare  {ms(t_pre):7.2f} ms")
        print(f"inferență     {ms(t_inf):7.2f} ms  (mediană {np.median(t_inf) * 1000:.2f} ms)")
        print(f"postprocesare {ms(t_post):7.2f} ms")
        print(f"total/cadru   {total:7.2f} ms  ->  {1000 / total:.1f} FPS")
        print(f"memorie maximă proces (VmHWM): {peak_memory_mb():.0f} MB")
        print(f"dimensiune model: {os.path.getsize(args.model) / 1024:.0f} kB")
        if gt_errs:
            print(f"AbsRel mediu față de LiDAR (0-{CONTRACT['max_depth']:.0f} m, la rezoluția "
                  f"modelului, {len(gt_errs)} imagini): {np.mean(gt_errs):.3f}")


if __name__ == "__main__":
    main()
