"""
Încărcarea datelor procesate (.npz) pentru antrenare și evaluare.
"""
import json
import os

import numpy as np
import tensorflow as tf

import config as C


def load_meta():
    with open(os.path.join(C.PROC_DIR, "meta.json")) as f:
        return json.load(f)


def load_split(split, with_teacher=False, with_gt=False):
    """Concatenează toate secvențele unui split. Returnează un dict de array-uri."""
    out = {"left": [], "right": [], "q": [], "mask": [], "gt": [], "drive": []}
    for drive in C.SPLITS[split]:
        name = C.drive_name(drive)
        path = os.path.join(C.PROC_DIR, name + ".npz")
        if not os.path.exists(path):
            print(f"ATENȚIE: lipsește {path}, sar peste")
            continue
        d = np.load(path)
        n = len(d["frames"])
        if with_teacher:
            t = np.load(os.path.join(C.PROC_DIR, name + C.TEACHER_SUFFIX))
            assert str(t["teacher_id"]) == C.TEACHER_ID, "Ținte de la alt teacher"
            assert np.array_equal(t["frames"], d["frames"][: len(t["frames"])])
            n = len(t["frames"])  # (la test rapid, teacher-ul poate avea mai puține cadre)
            out["q"].append(t["q"])
            out["mask"].append(t["mask"])
        out["left"].append(d["left"][:n])
        out["right"].append(d["right"][:n])
        if with_gt:
            out["gt"].append(d["gt"][:n])
        out["drive"] += [drive] * n
    return {k: (np.concatenate(v) if k != "drive" and len(v) else v) for k, v in out.items()}


def color_augment(left, right):
    """Aceeași modificare de culoare pe ambele imagini (geometria stereo rămâne validă)."""
    b = tf.shape(left)[0]
    bright = tf.random.uniform([b, 1, 1, 1], 0.8, 1.2)
    contrast = tf.random.uniform([b, 1, 1, 1], 0.8, 1.2)
    gamma = tf.random.uniform([b, 1, 1, 1], 0.9, 1.1)

    def f(x):
        m = tf.reduce_mean(x, [1, 2, 3], keepdims=True)
        x = (x - m) * contrast + m
        x = tf.clip_by_value(x * bright, 0, 1)
        return x ** gamma
    return f(left), f(right)


def train_batches(data, batch_size, seed):
    """Generator infinit de batch-uri amestecate (numpy -> tf.data)."""
    n = len(data["left"])
    has_t = len(data["q"]) > 0
    rng = np.random.default_rng(seed)

    def gen():
        while True:
            idx = rng.permutation(n)
            for i in range(0, n - batch_size + 1, batch_size):
                b = np.sort(idx[i:i + batch_size])
                if has_t:
                    q = data["q"][b][..., None].astype(np.float32)
                    m = data["mask"][b][..., None].astype(np.float32)
                else:  # baseline fără fișiere teacher: ținte goale (nefolosite)
                    q = np.full((len(b), C.IMG_H, C.IMG_W, 1), C.Q_MIN, np.float32)
                    m = np.zeros_like(q)
                yield data["left"][b], data["right"][b], q, m

    H, W = C.IMG_H, C.IMG_W
    sig = (tf.TensorSpec([batch_size, H, W, 3], tf.uint8),
           tf.TensorSpec([batch_size, H, W, 3], tf.uint8),
           tf.TensorSpec([batch_size, H, W, 1], tf.float32),
           tf.TensorSpec([batch_size, H, W, 1], tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(4)
