"""
FAZA 4 - Modelul student: MobileNetV3Small (encoder) + decoder mic tip U-Net.

Reguli pentru NPU (i.MX 8M Plus, delegate VX):
  * formă FIXĂ a intrării, batch 1 la export (fără dimensiuni dinamice);
  * doar operații simple: Conv2D, ReLU, Resizing (nearest), Concatenate;
  * nicio operație "exotică" (fără TransposeConv, fără resize bilinear cu
    align_corners, fără operații custom) -> altfel ajung pe CPU.
Ce operații rulează efectiv pe NPU se verifică pe placă (placa/benchmark.sh).

Ieșirea este s în [0, 1]; conversia în adâncime se face ÎN AFARA rețelei
(vezi decode_disparity), ca graful de pe NPU să rămână simplu.
"""
import numpy as np
import tensorflow as tf

import config as C

L = tf.keras.layers


def _skip_features(base):
    """
    Alege automat, pentru fiecare rezoluție (1/2, 1/4, 1/8, 1/16), ultimul strat
    al encoderului la acea rezoluție. Merge indiferent de numele straturilor
    din versiunea de Keras.
    """
    by_stride = {}
    for layer in base.layers:
        out = layer.output
        if isinstance(out, (list, tuple)) or len(out.shape) != 4:
            continue
        h = out.shape[1]
        if h is None or h == 1:
            continue  # straturile squeeze-excite (1 x 1) nu sunt hărți spațiale
        stride = C.IMG_H // h
        if stride in (2, 4, 8, 16):
            by_stride[stride] = out
    return [by_stride[s] for s in (2, 4, 8, 16)]


def conv_block(x, filters):
    x = L.Conv2D(filters, 3, padding="same", use_bias=True)(x)
    return L.ReLU()(x)


def build_student(minimalistic=True, imagenet=True):
    """
    minimalistic=True (IMPLICIT): MobileNetV3Small fără hard-swish și fără
    squeeze-excite. Graful int8 conține doar CONV_2D, DEPTHWISE_CONV_2D, ADD,
    CONCATENATION, RESIZE_NEAREST_NEIGHBOR, LOGISTIC (+1 MUL la intrare) ->
    potrivit pentru NPU. Varianta completă (minimalistic=False), cu greutăți
    ImageNet, dă eroare XNNPACK în TFLite 2.19 după cuantizarea int8 (blocul
    squeeze-excite expanded_conv_6), deci nu rulează nici pe CPU-ul plăcii.
    """
    inp = L.Input((C.IMG_H, C.IMG_W, 3), name="imagine")
    base = tf.keras.applications.MobileNetV3Small(
        input_tensor=inp, include_top=False,
        weights="imagenet" if imagenet else None,
        minimalistic=minimalistic,
        include_preprocessing=True,     # intrarea rămâne în [0, 255]
    )
    f2, f4, f8, f16 = _skip_features(base)
    x = base.output                     # rezoluție 1/32 (3 x 10)

    x = conv_block(x, 96)
    for skip, ch in ((f16, 64), (f8, 48), (f4, 32), (f2, 24)):
        # Resizing (nu UpSampling2D): în Keras 3, UpSampling2D devine RESHAPE+TILE
        # în TFLite, ceea ce strică XNNPACK pe CPU și nu e potrivit pentru NPU.
        # Resizing devine o singură operație RESIZE_NEAREST_NEIGHBOR.
        x = L.Resizing(skip.shape[1], skip.shape[2], interpolation="nearest")(x)
        x = L.Concatenate()([x, skip])
        x = conv_block(x, ch)
        x = conv_block(x, ch)
    out = L.Conv2D(1, 3, padding="same", activation="sigmoid", name="disp_s")(x)
    model = tf.keras.Model(inp, out, name="student")
    assert model.output_shape[1:3] == (C.OUT_H, C.OUT_W), model.output_shape
    return model


# ---------------------------------------------------------------------------
# Decodare ieșire (aceleași formule în TF, la antrenare, și în NumPy, pe placă)
# ---------------------------------------------------------------------------
LOG_QMIN, LOG_QMAX = np.log(C.Q_MIN), np.log(C.Q_MAX)


def s_to_q(s):
    """s in [0,1] -> disparitate normalizată q (TensorFlow)."""
    return tf.exp(LOG_QMIN + s * (LOG_QMAX - LOG_QMIN))


def q_to_s(q):
    return (tf.math.log(q) - LOG_QMIN) / (LOG_QMAX - LOG_QMIN)


def decode_disparity(s):
    """Versiunea NumPy (pentru evaluare și pe placă)."""
    return np.exp(LOG_QMIN + s.astype(np.float32) * (LOG_QMAX - LOG_QMIN))


if __name__ == "__main__":
    m = build_student()
    m.summary(line_length=110)
    print("Parametri:", m.count_params())
