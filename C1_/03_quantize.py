#!/usr/bin/env python3
"""
Paso 3: ONNX float32 -> TFLite int8 (cuantizacion completa) con onnx2tf.

La calibracion usa work/calib.npy (valores 0-1, RGB). onnx2tf aplica
(x - 0.5) / 0.5, que equivale al (x - 127.5) / 127.5 de ArcFace.
Uso: python 03_quantize.py [--per-tensor]
"""
import argparse
import subprocess
import sys

import onnxruntime as ort

from common import FP32_ONNX, WORK

ap = argparse.ArgumentParser()
ap.add_argument("--per-tensor", action="store_true",
                help="cuantizacion por tensor (por defecto: por canal)")
args = ap.parse_args()

if not (WORK / "calib.npy").exists():
    sys.exit("Falta work/calib.npy. Ejecuta primero 02_prepare_data.py")

name = ort.InferenceSession(str(FP32_ONNX), providers=["CPUExecutionProvider"]) \
    .get_inputs()[0].name
out_dir = WORK / "tflite"

cmd = [
    "onnx2tf", "-i", str(FP32_ONNX), "-o", str(out_dir),
    "-oiqt",
    "-tb", "tf_converter",                   # backend clasico de TensorFlow (mas estable)
    "-ois", f"{name}:1,3,112,112",          # fija batch = 1 (el ONNX lo trae dinamico)
    "-cind", name, str(WORK / "calib.npy"),
    "[[[[0.5,0.5,0.5]]]]", "[[[[0.5,0.5,0.5]]]]",
]
if args.per_tensor:
    cmd += ["-qt", "per-tensor"]

print(" ".join(cmd))
subprocess.run(cmd, check=True)

print("\nArchivos generados:")
for f in sorted(out_dir.glob("*.tflite")):
    print(f"  {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")
print("\nPara el Grove usaremos el *_full_integer_quant.tflite (entrada/salida int8).")
