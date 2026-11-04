"""
Pierderile de antrenare (se folosesc DOAR la antrenare, nu ajung pe placă).

  L = L_photo + W_SMOOTH * L_smooth + lambda_d * L_distill
  baseline : lambda_d = 0   (doar auto-supervizat, ca Monodepth)
  ghidat   : lambda_d > 0   (+ ținta teacher-ului)

Convenții: imaginile sunt float în [0, 1]; disparitatea q = d / W
(d în pixeli, W lățimea imaginii).
"""
import tensorflow as tf

import config as C
from student_model import s_to_q


# ---------------------------------------------------------------------------
# Reconstrucția imaginii stângi din cea dreaptă
# ---------------------------------------------------------------------------
def warp_right_to_left(right, disp_px):
    """
    Pentru o pereche stereo rectificată, pixelul (x, y) din stânga se găsește
    în dreapta la (x - d, y). Eșantionăm imaginea dreaptă acolo (interpolare
    liniară pe orizontală), obținând o "imagine stângă reconstruită".
    right: [B,H,W,3], disp_px: [B,H,W,1] -> (reconstruită [B,H,W,3], mască [B,H,W,1])
    """
    W = tf.shape(right)[2]
    Wf = tf.cast(W, tf.float32)
    xs = tf.cast(tf.range(W), tf.float32)[None, None, :]
    x_src = xs - disp_px[..., 0]                         # [B,H,W]
    x0 = tf.floor(x_src)
    w1 = x_src - x0
    i0 = tf.clip_by_value(tf.cast(x0, tf.int32), 0, W - 1)
    i1 = tf.clip_by_value(tf.cast(x0, tf.int32) + 1, 0, W - 1)
    p0 = tf.gather(right, i0, axis=2, batch_dims=2)
    p1 = tf.gather(right, i1, axis=2, batch_dims=2)
    warped = p0 * (1.0 - w1[..., None]) + p1 * w1[..., None]
    valid = tf.cast((x_src >= 0.0) & (x_src <= Wf - 1.0), tf.float32)[..., None]
    return warped, valid


def ssim(x, y):
    """SSIM pe ferestre 3x3 (ca în Monodepth2). Returnează [B,H,W,3]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = [[0, 0], [1, 1], [1, 1], [0, 0]]
    x = tf.pad(x, pad, mode="REFLECT")
    y = tf.pad(y, pad, mode="REFLECT")
    pool = lambda t: tf.nn.avg_pool2d(t, 3, 1, "VALID")  # noqa: E731
    mu_x, mu_y = pool(x), pool(y)
    sx = pool(x * x) - mu_x ** 2
    sy = pool(y * y) - mu_y ** 2
    sxy = pool(x * y) - mu_x * mu_y
    n = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
    d = (mu_x ** 2 + mu_y ** 2 + C1) * (sx + sy + C2)
    return tf.clip_by_value((1 - n / d) / 2, 0, 1)


def photometric_error(a, b):
    """Eroare per pixel: 0.85 * SSIM + 0.15 * L1 -> [B,H,W,1]."""
    l1 = tf.reduce_mean(tf.abs(a - b), -1, keepdims=True)
    ss = tf.reduce_mean(ssim(a, b), -1, keepdims=True)
    return C.SSIM_ALPHA * ss + (1 - C.SSIM_ALPHA) * l1


def masked_mean(x, m):
    return tf.reduce_sum(x * m) / tf.maximum(tf.reduce_sum(m), 1.0)


# ---------------------------------------------------------------------------
# Netezime (permite salturi doar unde imaginea are muchii)
# ---------------------------------------------------------------------------
def smoothness(q, img):
    q = q / (tf.reduce_mean(q, [1, 2, 3], keepdims=True) + 1e-7)
    dq_x = tf.abs(q[:, :, :-1] - q[:, :, 1:])
    dq_y = tf.abs(q[:, :-1] - q[:, 1:])
    di_x = tf.reduce_mean(tf.abs(img[:, :, :-1] - img[:, :, 1:]), -1, keepdims=True)
    di_y = tf.reduce_mean(tf.abs(img[:, :-1] - img[:, 1:]), -1, keepdims=True)
    return (tf.reduce_mean(dq_x * tf.exp(-di_x)) + tf.reduce_mean(dq_y * tf.exp(-di_y)))


# ---------------------------------------------------------------------------
# Pierderea totală
# ---------------------------------------------------------------------------
def total_loss(s_pred, left, right, q_teacher=None, mask_teacher=None, w_distill=0.0):
    """
    s_pred: ieșirea rețelei [B,48,160,1]; left/right: [B,96,320,3] în [0,1].
    q_teacher/mask_teacher: [B,96,320,1] (doar în modul ghidat).
    """
    q_small = s_to_q(s_pred)
    q_full = tf.image.resize(q_small, (C.IMG_H, C.IMG_W), method="bilinear")
    disp_px = q_full * C.IMG_W

    recon, valid = warp_right_to_left(right, disp_px)
    l_photo = masked_mean(photometric_error(recon, left), valid)

    left_small = tf.image.resize(left, (C.OUT_H, C.OUT_W), method="area")
    l_smooth = smoothness(q_small, left_small)

    loss = l_photo + C.W_SMOOTH * l_smooth
    logs = {"photo": l_photo, "smooth": l_smooth}

    if q_teacher is not None and w_distill > 0:
        # L1 în spațiu logaritmic: aceeași scară ca ieșirea rețelei, iar
        # obiectele îndepărtate (disparitate mică) contează la fel de mult.
        qt = tf.clip_by_value(q_teacher, C.Q_MIN, C.Q_MAX)
        l_distill = masked_mean(tf.abs(tf.math.log(q_full) - tf.math.log(qt)), mask_teacher)
        loss = loss + w_distill * l_distill
        logs["distill"] = l_distill
    logs["total"] = loss
    return loss, logs
