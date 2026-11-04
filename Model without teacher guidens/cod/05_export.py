"""
FAZA 8 - Export pentru NPU: Keras -> TFLite float -> TFLite int8.

Generează în EXPORT_DIR, pentru modelul ales:
  student_<nume>_float.tflite      referință (rulează doar pe CPU)
  student_<nume>_int8.tflite       int8 complet, per-canal (implicit TFLite)
  student_<nume>_int8_pt.tflite    int8 complet, per-tensor (NPU-ul i.MX 8M Plus
                                   e optimizat pentru per-tensor -> comparăm viteza)
  ref_input.npy / ref_output_<nume>_int8.npy   pentru verificarea pe placă
  export_<nume>.json               dimensiuni, cuantizare, diferențe față de Keras
  ops_<nume>_int8.txt              lista operațiilor din graf

Calibrarea folosește imagini REALE de antrenament (niciodată de test).
TensorFlow trebuie să fie 2.19.x, ca TFLite 2.19.0 de pe placă.

    python 05_export.py --nume baseline
    python 05_export.py --nume ghidat
"""
import argparse
import contextlib
import io
import json
import os

import numpy as np
import tensorflow as tf

import config as C
import dataset
import tflite_runner

if not tf.__version__.startswith("2.19"):
    print(f"ATENȚIE: TensorFlow {tf.__version__}; placa are TFLite 2.19.0. "
          "Instalează tensorflow==2.19.0 ca să eviți erori de versiune a operațiilor.")


def convert(model, calib=None, per_tensor=False):
    # Intrare cu batch FIX = 1 (fără dimensiuni dinamice -> mai simplu pentru NPU)
    inp = tf.keras.Input(batch_shape=(1, C.IMG_H, C.IMG_W, 3), name="imagine")
    fixed = tf.keras.Model(inp, model(inp, training=False))
    conv = tf.lite.TFLiteConverter.from_keras_model(fixed)
    if calib is not None:
        def rep():
            for img in calib:
                yield [img[None].astype(np.float32)]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        if per_tensor:
            conv._experimental_disable_per_channel = True
    return conv.convert()


def list_ops(tfl_bytes):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tf.lite.experimental.Analyzer.analyze(model_content=tfl_bytes)
    text = buf.getvalue()
    ops = {}
    for line in text.splitlines():
        if "Op#" in line:
            name = line.split()[1].split("(")[0]
            ops[name] = ops.get(name, 0) + 1
    return text, ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nume", required=True, help="folderul din RUNS_DIR (ex. baseline, ghidat)")
    ap.add_argument("--checkpoint", default="best.keras")
    ap.add_argument("--n_calib", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(C.EXPORT_DIR, exist_ok=True)
    model = tf.keras.models.load_model(os.path.join(C.RUNS_DIR, args.nume, args.checkpoint),
                                       compile=False)
    train = dataset.load_split("train")
    val = dataset.load_split("val")
    rng = np.random.default_rng(C.SEED)
    calib = train["left"][rng.choice(len(train["left"]),
                                     min(args.n_calib, len(train["left"])), replace=False)]
    check = val["left"][: min(50, len(val["left"]))]          # imagini de verificare

    y_keras = model.predict(check.astype(np.float32), batch_size=16, verbose=0)
    variants = {
        "float": convert(model),
        "int8": convert(model, calib),
        "int8_pt": convert(model, calib, per_tensor=True),
    }

    info = {"tensorflow": tf.__version__, "params": int(model.count_params()), "variante": {}}
    for kind, tfl in variants.items():
        path = os.path.join(C.EXPORT_DIR, f"student_{args.nume}_{kind}.tflite")
        with open(path, "wb") as f:
            f.write(tfl)
        it = tflite_runner.make_interpreter(tfl)
        y = tflite_runner.predict_all(it, check)
        diff = np.abs(y - y_keras)
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        info["variante"][kind] = {
            "fisier": os.path.basename(path), "kB": round(len(tfl) / 1024, 1),
            "intrare": [str(np.dtype(i["dtype"])), list(map(float, i["quantization"]))],
            "iesire": [str(np.dtype(o["dtype"])), list(map(float, o["quantization"]))],
            "dif_medie_vs_keras": float(diff.mean()), "dif_max_vs_keras": float(diff.max()),
        }
        print(f"{kind:8s} {len(tfl) / 1024:8.1f} kB | diferență s față de Keras: "
              f"medie {diff.mean():.4f}, max {diff.max():.4f}")
        if kind == "int8":
            text, ops = list_ops(tfl)
            with open(os.path.join(C.EXPORT_DIR, f"ops_{args.nume}_int8.txt"), "w") as f:
                f.write(text)
            info["operatii_int8"] = ops
            print("Operații în graful int8:", ops)
            np.save(os.path.join(C.EXPORT_DIR, "ref_input.npy"), check[:1])
            np.save(os.path.join(C.EXPORT_DIR, f"ref_output_{args.nume}_int8.npy"), y[:1])

    with open(os.path.join(C.EXPORT_DIR, f"export_{args.nume}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("Fișiere salvate în", C.EXPORT_DIR)


if __name__ == "__main__":
    main()
