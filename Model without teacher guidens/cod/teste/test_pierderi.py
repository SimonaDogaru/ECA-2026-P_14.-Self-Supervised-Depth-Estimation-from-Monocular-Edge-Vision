"""
Teste rapide (fără date KITTI) pentru pierderi și model. Rulează din folderul cod/:
    python teste/test_pierderi.py
Dacă un test pică, NU porni antrenarea lungă: ceva e greșit la geometrie.
"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
import losses  # noqa: E402
from student_model import build_student, s_to_q, q_to_s, decode_disparity  # noqa: E402


def test_directie_warp():
    """Stânga(x) = Dreapta(x - d): cu d corect reconstrucția e perfectă, cu -d nu."""
    rng = np.random.default_rng(0)
    right = rng.uniform(0, 1, (1, C.IMG_H, C.IMG_W, 3)).astype(np.float32)
    d = 7
    left = np.zeros_like(right)
    left[:, :, d:] = right[:, :, :-d]                  # obiectele "se mută" spre dreapta în stânga
    disp = np.full((1, C.IMG_H, C.IMG_W, 1), d, np.float32)
    rec, valid = losses.warp_right_to_left(tf.constant(right), tf.constant(disp))
    v = valid.numpy()[..., 0] > 0
    err_ok = np.abs(rec.numpy() - left)[v].mean()
    rec_bad, vb = losses.warp_right_to_left(tf.constant(right), tf.constant(-disp))
    err_bad = np.abs(rec_bad.numpy() - left)[vb.numpy()[..., 0] > 0].mean()
    assert err_ok < 1e-5, err_ok
    assert err_bad > 0.1, err_bad
    assert not v[:, :, :d].any(), "pixelii fără corespondent trebuie mascați"
    print(f"  warp: eroare cu d corect {err_ok:.2e}, cu d inversat {err_bad:.3f}")


def test_warp_subpixel():
    """Disparitate fracționară -> interpolare liniară corectă și gradient spre disparitate."""
    x = np.arange(C.IMG_W, dtype=np.float32)
    right = np.tile(x[None, None, :, None], (1, C.IMG_H, 1, 3)) / C.IMG_W
    disp = tf.Variable(np.full((1, C.IMG_H, C.IMG_W, 1), 3.5, np.float32))
    with tf.GradientTape() as t:
        rec, valid = losses.warp_right_to_left(tf.constant(right), disp)
        loss = tf.reduce_sum(rec * valid)
    expected = (x[10] - 3.5) / C.IMG_W
    assert abs(rec.numpy()[0, 0, 10, 0] - expected) < 1e-5
    g = t.gradient(loss, disp)
    assert g is not None and np.isfinite(g.numpy()).all() and np.abs(g.numpy()).sum() > 0
    print("  warp subpixel + gradient: OK")


def test_ssim_identic():
    a = tf.random.uniform((2, 32, 32, 3))
    assert float(tf.reduce_max(losses.photometric_error(a, a))) < 1e-4
    print("  eroare fotometrică imagini identice ~ 0: OK")


def test_decodare():
    s = np.linspace(0, 1, 11, dtype=np.float32)
    q_tf = s_to_q(tf.constant(s)).numpy()
    assert np.allclose(q_tf, decode_disparity(s), rtol=1e-5)          # TF == NumPy (placă)
    assert np.allclose(q_to_s(tf.constant(q_tf)).numpy(), s, atol=1e-5)
    assert abs(q_tf[0] - C.Q_MIN) < 1e-7 and abs(q_tf[-1] - C.Q_MAX) < 1e-6
    print("  decodare s <-> q (TF și NumPy identice): OK")


def test_pas_antrenare():
    """Un pas complet: forme corecte, pierderi finite, gradient pentru toate straturile."""
    model = build_student(imagenet=False)
    b = 2
    left = tf.random.uniform((b, C.IMG_H, C.IMG_W, 3))
    right = tf.random.uniform((b, C.IMG_H, C.IMG_W, 3))
    q_t = tf.fill((b, C.IMG_H, C.IMG_W, 1), 0.05)
    m_t = tf.ones((b, C.IMG_H, C.IMG_W, 1))
    with tf.GradientTape() as t:
        s = model(left * 255.0, training=True)
        loss, logs = losses.total_loss(s, left, right, q_t, m_t, w_distill=1.0)
    assert s.shape == (b, C.OUT_H, C.OUT_W, 1)
    grads = t.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in grads), "un strat nu primește gradient"
    assert all(np.isfinite(g.numpy()).all() for g in grads)
    assert set(logs) == {"photo", "smooth", "distill", "total"}
    _, logs0 = losses.total_loss(s, left, right, q_t, m_t, w_distill=0.0)
    assert "distill" not in logs0, "baseline-ul nu trebuie să folosească teacher-ul"
    print(f"  pas antrenare: pierdere {float(loss):.3f}, {len(grads)} gradienți finiți: OK")


if __name__ == "__main__":
    for f in [test_directie_warp, test_warp_subpixel, test_ssim_identic,
              test_decodare, test_pas_antrenare]:
        print(f.__name__)
        f()
    print("TOATE TESTELE AU TRECUT")
