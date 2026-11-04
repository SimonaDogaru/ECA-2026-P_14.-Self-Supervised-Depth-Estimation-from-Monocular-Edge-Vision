"""
FAZELE 5 și 6 - Antrenarea studentului.

Același script, aceeași arhitectură, aceleași date și setări; diferă DOAR
pierderea teacher-ului:
    python 03_antrenare.py --mod baseline     # fără teacher (sarcina Simonei)
    python 03_antrenare.py --mod ghidat       # cu teacher

La fiecare epocă se evaluează pe validare (AbsRel față de LiDAR) și se
păstrează cel mai bun model: RUNS_DIR/<mod>/best.keras (+ last.keras).
Setul de TEST nu se folosește aici niciodată.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import tensorflow as tf

import config as C
import dataset
import losses
import metrics
from student_model import build_student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", choices=["baseline", "ghidat"], required=True)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--steps_per_epoch", type=int, default=None,
                    help="implicit: o trecere completă prin datele de antrenare")
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--w_distill", type=float, default=C.W_DISTILL)
    ap.add_argument("--complet", action="store_true",
                    help="MobileNetV3Small complet (cu SE/hard-swish); implicit: minimalistic")
    ap.add_argument("--name", default=None, help="nume folder rezultat (implicit = mod)")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(C.SEED)   # aceeași inițializare pentru ambele moduri
    guided = args.mod == "ghidat"
    w_distill = args.w_distill if guided else 0.0
    run_dir = os.path.join(C.RUNS_DIR, args.name or args.mod)
    os.makedirs(run_dir, exist_ok=True)

    # Încărcăm teacher-ul și pentru baseline doar ca să folosim EXACT aceleași
    # cadre; baseline-ul nu îl folosește în pierdere (w_distill = 0).
    try:
        train = dataset.load_split("train", with_teacher=True)
    except FileNotFoundError:
        if guided:
            raise SystemExit("Lipsesc țintele teacher-ului: rulează întâi 02_teacher.py")
        print("Fișierele teacher lipsesc -> baseline pe toate cadrele, fără teacher")
        train = dataset.load_split("train")
    val = dataset.load_split("val", with_gt=True)
    meta = dataset.load_meta()
    n_train = len(train["left"])
    steps = args.steps_per_epoch or max(1, n_train // args.batch)
    print(f"Mod: {args.mod} | cadre train: {n_train} | val: {len(val['left'])} | "
          f"pași/epocă: {steps} | w_distill: {w_distill}")

    model = build_student(minimalistic=not args.complet)
    total_steps = steps * args.epochs
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [int(0.75 * total_steps)], [C.LR, C.LR / 10])
    opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def train_step(left_u8, right_u8, q_t, m_t):
        left = tf.cast(left_u8, tf.float32) / 255.0
        right = tf.cast(right_u8, tf.float32) / 255.0
        left_aug, _ = dataset.color_augment(left, right)
        with tf.GradientTape() as tape:
            s = model(left_aug * 255.0, training=True)   # rețeaua primește [0, 255]
            loss, logs = losses.total_loss(s, left, right, q_t, m_t, w_distill)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return logs

    def predict(images, bs=32):
        out = []
        for i in range(0, len(images), bs):
            out.append(model(images[i:i + bs].astype(np.float32), training=False).numpy())
        return np.concatenate(out)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "w_distill": w_distill, "n_train": n_train,
                   "img": [C.IMG_H, C.IMG_W], "out": [C.OUT_H, C.OUT_W],
                   "q_range": [C.Q_MIN, C.Q_MAX], "lr": C.LR, "seed": C.SEED,
                   "w_smooth": C.W_SMOOTH, "tf": tf.__version__}, f, indent=2)

    log_path = os.path.join(run_dir, "log.csv")
    best = np.inf
    it = iter(dataset.train_batches(train, args.batch, C.SEED))
    with open(log_path, "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["epoca", "photo", "smooth", "distill", "total", "val_abs_rel",
                     "val_rmse", "val_a1", "secunde"])
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            acc = {}
            for _ in range(steps):
                batch = next(it)
                logs = train_step(batch[0], batch[1], batch[2], batch[3])
                for k, v in logs.items():
                    acc[k] = acc.get(k, 0.0) + float(v) / steps
            val_m = metrics.evaluate(metrics.s_batch_to_q(predict(val["left"])), val["gt"], meta)
            dt = time.time() - t0
            print(f"Epoca {ep}/{args.epochs}  photo {acc['photo']:.4f}  "
                  f"distill {acc.get('distill', 0):.4f}  val AbsRel {val_m['abs_rel']:.4f}  "
                  f"val RMSE {val_m['rmse']:.3f}  δ1 {val_m['a1']:.3f}  ({dt:.0f}s)")
            wr.writerow([ep, acc["photo"], acc["smooth"], acc.get("distill", 0), acc["total"],
                         val_m["abs_rel"], val_m["rmse"], val_m["a1"], round(dt)])
            fcsv.flush()
            model.save(os.path.join(run_dir, "last.keras"))
            if val_m["abs_rel"] < best:
                best = val_m["abs_rel"]
                model.save(os.path.join(run_dir, "best.keras"))
                print(f"  -> model nou cel mai bun (val AbsRel {best:.4f})")
    print("Gata. Rezultate în", run_dir)


if __name__ == "__main__":
    main()
