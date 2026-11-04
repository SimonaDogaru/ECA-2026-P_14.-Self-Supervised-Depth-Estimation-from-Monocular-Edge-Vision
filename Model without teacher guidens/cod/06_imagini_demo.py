"""
FAZA 10 (pregătire pe PC) - Salvează câteva imagini de TEST ca PNG pentru demo-ul de pe placă.

    python 06_imagini_demo.py --n 30
Rezultat: EXPORT_DIR/demo_imagini/*.png (deja la 320 x 96, ca intrarea modelului).
"""
import argparse
import os

import numpy as np
from PIL import Image

import config as C
import dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    data = dataset.load_split("test")
    out = os.path.join(C.EXPORT_DIR, "demo_imagini")
    os.makedirs(out, exist_ok=True)
    idx = np.linspace(0, len(data["left"]) - 1, min(args.n, len(data["left"]))).astype(int)
    for k, i in enumerate(idx):
        Image.fromarray(data["left"][i]).save(os.path.join(out, f"{k:03d}_drive{data['drive'][i]}.png"))
    print(f"{len(idx)} imagini salvate în {out}")


if __name__ == "__main__":
    main()
