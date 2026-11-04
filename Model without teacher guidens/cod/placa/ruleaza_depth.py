#!/usr/bin/env python3
"""
FAZELE 9-10 pe placă: inferență student pe NPU + demo (hartă de adâncime colorată).

Scriptul e independent (doar numpy, Pillow, tflite_runtime; OpenCV opțional).
Copiază pe placă: acest fișier + modelul .tflite (+ imagini / fișierele ref_*.npy).

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

# ---- Contractul modelului: IDENTIC cu cod/config.py ----
IMG_H, IMG_W = 96, 320
Q_MIN, Q_MAX = 0.001, 0.3
# Camera KITTI (2011_09_26): fx [px], baseline [m], lățime originală [px]
FX, BASELINE, ORIG_W = 721.5377, 0.5327, 1242
DELEGATE = "/usr/lib/libvx_delegate.so"


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
    """La fel ca la antrenare: RGB, redimensionare biliniară la 320 x 96, valori [0, 255]."""
    img = img_pil.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None]


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
    it.set_tensor(i["index"], quantize_input(x, i))
    it.invoke()
    return dequantize_output(it.get_tensor(o["index"]), o)[0, ..., 0]   # s, [48,160]


def s_to_depth(s):
    q = np.exp(np.log(Q_MIN) + s * (np.log(Q_MAX) - np.log(Q_MIN)))
    return FX * BASELINE / (q * ORIG_W)


def turbo(x):
    """Paleta 'turbo' (aproximare polinomială Google), x în [0,1] -> RGB uint8."""
    x = np.clip(x, 0, 1)[..., None]
    c = np.concatenate([
        0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943)))),
        0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604)))),
        0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973)))),
    ], -1)
    return (np.clip(c, 0, 1) * 255).astype(np.uint8)


def visualize(x, s):
    """Imaginea de intrare deasupra, harta de adâncime dedesubt (roșu = aproape)."""
    lo, hi = np.percentile(s, 2), np.percentile(s, 98)
    color = turbo((s - lo) / max(hi - lo, 1e-6))
    color = np.asarray(Image.fromarray(color).resize((IMG_W, IMG_H), Image.NEAREST))
    return np.concatenate([x[0].astype(np.uint8), color], 0)


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
            yield f"cam_{int(time.time() * 1000)}", Image.fromarray(frame[..., ::-1])
    else:
        paths = [source] if os.path.isfile(source) else sorted(
            glob.glob(os.path.join(source, "*.png")) + glob.glob(os.path.join(source, "*.jpg")))
        for p in paths:
            yield os.path.splitext(os.path.basename(p))[0], Image.open(p)


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

    t0 = time.perf_counter()
    it = make_interpreter(args.model, args.npu)
    t_init = time.perf_counter() - t0
    dev = "NPU (VX delegate)" if args.npu else "CPU (4 fire)"
    print(f"Model: {args.model} | rulare pe: {dev} | inițializare: {t_init * 1000:.0f} ms")

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
    for n, (name, img) in enumerate(frames_from(args.intrare)):
        a = time.perf_counter()
        x = preprocess(img)
        b = time.perf_counter()
        s = infer(it, x)
        c = time.perf_counter()
        vis = visualize(x, s)
        depth = s_to_depth(s)
        d = time.perf_counter()
        if n == 0:
            print(f"Primul cadru (încălzire NPU, exclus din medie): {(c - b) * 1000:.1f} ms")
        else:
            t_pre.append(b - a)
            t_inf.append(c - b)
            t_post.append(d - c)
        h, w = depth.shape
        print(f"{name}: inferență {(c - b) * 1000:.1f} ms | adâncime centru ~ {depth[h // 2, w // 2]:.1f} m")
        if args.afiseaza:
            import cv2
            fps = 1.0 / max(np.mean(t_pre[-10:] + t_inf[-10:] + t_post[-10:]) * 3, 1e-6) if t_inf else 0
            big = cv2.resize(vis[..., ::-1], (IMG_W * 3, IMG_H * 6), interpolation=cv2.INTER_NEAREST)
            cv2.putText(big, f"{dev}  {fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Adancime monoculara - student", big)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            Image.fromarray(vis).resize((IMG_W * 2, IMG_H * 4), Image.NEAREST).save(
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


if __name__ == "__main__":
    main()
