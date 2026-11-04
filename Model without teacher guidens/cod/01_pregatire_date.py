"""
FAZA 2 - Pregătirea datelor KITTI.

Pentru fiecare secvență din config.SPLITS:
  1. descarcă arhiva și extrage camera stânga/dreapta (+ LiDAR pentru val/test);
  2. redimensionează imaginile la contractul modelului (96 x 320);
  3. pentru val/test calculează adâncimea reală din LiDAR (rezoluție originală);
  4. salvează totul într-un singur fișier: PROC_DIR/<secvență>.npz

Rulare (Colab):
    python 01_pregatire_date.py                 # toate secvențele
    python 01_pregatire_date.py --splits test   # doar test
    python 01_pregatire_date.py --max_frames 10 # test rapid
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

import config as C
import kitti


def load_resized(path):
    """Aceeași funcție trebuie folosită și pe placă (vezi placa/ruleaza_depth.py)."""
    img = Image.open(path).convert("RGB").resize((C.IMG_W, C.IMG_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def process_drive(split, drive, max_frames=None):
    name = C.drive_name(drive)
    out_path = os.path.join(C.PROC_DIR, name + ".npz")
    if os.path.exists(out_path):
        print(f"[{split}] {name}: există deja, sar peste")
        return
    need_gt = split in ("val", "test")
    folder = kitti.download_drive(drive, need_velodyne=need_gt)

    lefts = sorted(glob.glob(os.path.join(folder, "image_02", "data", "*.png")))
    lefts = lefts[:: C.FRAME_STEP[split]]
    if max_frames:
        lefts = lefts[:max_frames]

    L, R, G, frames = [], [], [], []
    for lp in lefts:
        frame = os.path.splitext(os.path.basename(lp))[0]
        rp = lp.replace("image_02", "image_03")
        if not os.path.exists(rp):
            continue  # cadru fără pereche -> îl ignorăm
        L.append(load_resized(lp))
        R.append(load_resized(rp))
        frames.append(int(frame))
        if need_gt:
            vp = os.path.join(folder, "velodyne_points", "data", frame + ".bin")
            G.append(kitti.velodyne_to_depth(vp).astype(np.float16))

    data = {"left": np.stack(L), "right": np.stack(R), "frames": np.array(frames)}
    if need_gt:
        data["gt"] = np.stack(G)
    os.makedirs(C.PROC_DIR, exist_ok=True)
    np.savez_compressed(out_path, **data)
    print(f"[{split}] {name}: {len(frames)} cadre salvate -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    kitti.download_calib()
    fx, B, w, h = kitti.camera_params()
    meta = {"fx": fx, "baseline": B, "orig_w": w, "orig_h": h,
            "img_h": C.IMG_H, "img_w": C.IMG_W, "splits": C.SPLITS}
    os.makedirs(C.PROC_DIR, exist_ok=True)
    with open(os.path.join(C.PROC_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Calibrare: fx={fx:.2f}px, baseline={B:.4f}m, imagine {w}x{h}")

    for split in args.splits:
        for drive in C.SPLITS[split]:
            process_drive(split, drive, args.max_frames)


if __name__ == "__main__":
    main()
