"""
FAZA 7 - Evaluare pe setul de TEST (o singură dată, la final).

Compară, cu aceleași cadre și aceeași mască LiDAR:
  * teacher-ul (harta completă salvată la 96 x 320, evaluată exact ca studentul);
  * studenții Keras (baseline, ghidat);
  * modelele .tflite exportate (float, int8) -> efectul cuantizării.

    python 04_evaluare.py --modele baseline ghidat --tflite student_baseline_int8.tflite student_ghidat_int8.tflite
Rezultatul: ROOT/rezultate_test.md și imagini comparative în ROOT/figuri/.
"""
import argparse
import os

import numpy as np

import config as C
import dataset
import metrics
import tflite_runner


def save_figure(path, left, maps, titles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 1 + len(maps)
    fig, ax = plt.subplots(n, 1, figsize=(8, 1.6 * n))
    ax[0].imshow(left)
    ax[0].set_title("Imagine (intrare)")
    for a, m, t in zip(ax[1:], maps, titles):
        a.imshow(m, cmap="magma", vmin=0, vmax=np.percentile(maps[0], 95))
        a.set_title(t)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", nargs="*", default=["baseline", "ghidat"])
    ap.add_argument("--tflite", nargs="*", default=[])
    ap.add_argument("--checkpoint", default="best.keras")
    ap.add_argument("--fara_teacher", action="store_true")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    meta = dataset.load_meta()
    with_t = not args.fara_teacher
    try:
        data = dataset.load_split(args.split, with_teacher=with_t, with_gt=True)
    except FileNotFoundError:
        with_t = False
        data = dataset.load_split(args.split, with_gt=True)
    print(f"Evaluare pe {args.split}: {len(data['left'])} cadre")

    rows, preds = [], {}
    if with_t:
        q_t = [q.astype(np.float32) for q in data["q"]]
        preds["teacher (GMStereo)"] = q_t
        rows.append(("teacher GMStereo (stereo, GPU)", q_t))

    if args.modele:
        import tensorflow as tf
        for name in args.modele:
            path = os.path.join(C.RUNS_DIR, name, args.checkpoint)
            if not os.path.exists(path):
                print("lipsește", path)
                continue
            model = tf.keras.models.load_model(path, compile=False)
            s = model.predict(data["left"].astype(np.float32), batch_size=32, verbose=0)
            q = metrics.s_batch_to_q(s)
            preds[f"student {name}"] = q
            rows.append((f"student {name} (Keras float)", q))

    for fname in args.tflite:
        it = tflite_runner.make_interpreter(os.path.join(C.EXPORT_DIR, fname))
        q = metrics.s_batch_to_q(tflite_runner.predict_all(it, data["left"]))
        rows.append((fname, q))
        preds[fname] = q

    lines = [f"# Rezultate pe {args.split} ({len(data['left'])} cadre, LiDAR, Garg crop, 80 m)", "",
             "Scară metrică din geometria stereo (fără median scaling):", "", metrics.TABLE_HEADER]
    res_all = []
    for name, q in rows:
        m = metrics.evaluate(q, data["gt"], meta)
        res_all.append((name, q))
        lines.append(metrics.format_row(name, m))
        print(metrics.format_row(name, m))
    lines += ["", "Cu median scaling (doar pentru comparație cu articolele monoculare):", "",
              metrics.TABLE_HEADER]
    for name, q in res_all:
        lines.append(metrics.format_row(name, metrics.evaluate(q, data["gt"], meta, True)))

    out = os.path.join(C.ROOT, f"rezultate_{args.split}.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Tabel salvat:", out)

    fig_dir = os.path.join(C.ROOT, "figuri")
    os.makedirs(fig_dir, exist_ok=True)
    for k in np.linspace(0, len(data["left"]) - 1, min(5, len(data["left"]))).astype(int):
        maps = [preds[n][k] for n in preds]
        save_figure(os.path.join(fig_dir, f"{args.split}_{k:04d}.png"),
                    data["left"][k], maps, list(preds))
    print("Figuri salvate în", fig_dir)


if __name__ == "__main__":
    main()
