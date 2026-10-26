"""Utilidades compartidas por los scripts de la etapa C1."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
WORK = ROOT / "work"
FP32_ONNX = MODELS / "embedding_fp32.onnx"


def normalize_rgb(rgb_u8):
    """uint8 RGB (..., 112, 112, 3) -> float32 en [-1, 1], igual que ArcFace."""
    return (rgb_u8.astype(np.float32) - 127.5) / 127.5


def l2norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)
