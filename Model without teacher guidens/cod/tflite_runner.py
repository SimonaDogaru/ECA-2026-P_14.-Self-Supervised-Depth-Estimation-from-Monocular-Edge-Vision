"""Rulare model .tflite pe PC (pentru verificare). Pe placă se folosește placa/ruleaza_depth.py."""
import numpy as np


def make_interpreter(path_or_bytes):
    # Preferăm interpretorul din TensorFlow 2.19 = aceeași versiune ca pe placă.
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except (ImportError, AttributeError):
        from ai_edge_litert.interpreter import Interpreter
    if isinstance(path_or_bytes, (bytes, bytearray)):
        it = Interpreter(model_content=bytes(path_or_bytes))
    else:
        it = Interpreter(model_path=path_or_bytes)
    it.allocate_tensors()
    return it


def run(it, x_float):
    """x_float: [1,H,W,3] în [0,255]. Returnează ieșirea dequantizată (float32)."""
    i = it.get_input_details()[0]
    o = it.get_output_details()[0]
    if np.issubdtype(i["dtype"], np.integer):
        s, z = i["quantization"]
        info = np.iinfo(i["dtype"])
        x = np.clip(np.round(x_float / s) + z, info.min, info.max).astype(i["dtype"])
    else:
        x = x_float.astype(i["dtype"])
    it.set_tensor(i["index"], x)
    it.invoke()
    y = it.get_tensor(o["index"])
    if np.issubdtype(o["dtype"], np.integer):
        s, z = o["quantization"]
        y = s * (y.astype(np.float32) - z)
    return y.astype(np.float32)


def predict_all(it, images_u8):
    return np.concatenate([run(it, img[None].astype(np.float32)) for img in images_u8])
