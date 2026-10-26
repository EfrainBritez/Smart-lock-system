#!/usr/bin/env python3
"""
Paso 1: ubicar el modelo de embeddings (MobileFaceNet de InsightFace).

El paquete buffalo_s trae varios .onnx; el de reconocimiento es el que tiene
entrada 112x112 y salida de un vector. Lo copiamos a models/.
Uso: python 01_get_model.py [buffalo_s | buffalo_sc | buffalo_l]
"""
import shutil
import sys
from pathlib import Path

import onnxruntime as ort
from insightface.app import FaceAnalysis

from common import FP32_ONNX, MODELS

pack = sys.argv[1] if len(sys.argv) > 1 else "buffalo_s"
FaceAnalysis(name=pack, providers=["CPUExecutionProvider"])  # descarga si falta
pack_dir = Path.home() / ".insightface" / "models" / pack

found = None
for f in sorted(pack_dir.glob("*.onnx")):
    s = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
    inp, out = s.get_inputs()[0], s.get_outputs()[0]
    if list(inp.shape[2:]) == [112, 112] and len(out.shape) == 2:
        found = (f, inp, out)
        break

if not found:
    sys.exit(f"No encontre un modelo de reconocimiento en {pack_dir}")

f, inp, out = found
MODELS.mkdir(exist_ok=True)
shutil.copy(f, FP32_ONNX)
print(f"Modelo:   {f}")
print(f"Copiado:  {FP32_ONNX} ({f.stat().st_size / 1e6:.1f} MB)")
print(f"Entrada:  {inp.name} {inp.shape}")
print(f"Salida:   {out.name} {out.shape}  -> embedding de {out.shape[-1]} dimensiones")
