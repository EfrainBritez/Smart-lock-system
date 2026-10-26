#!/usr/bin/env python3
"""
Paso 2: preparar datos.
  - Detecta y alinea rostros a 112x112 (igual que InsightFace en la PC)
  - Guarda los recortes en work/crops/<persona>/ (para evaluar)
  - Genera work/calib.npy (para cuantizar a int8)

Estructura esperada de fotos:  dataset/<persona>/foto1.jpg, foto2.jpg, ...

Uso:
  python 02_prepare_data.py --images dataset
  python 02_prepare_data.py --lfw        # descarga fotos publicas (LFW) de ejemplo
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from common import WORK

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def download_lfw(out_dir, max_people, per_person):
    from sklearn.datasets import fetch_lfw_people

    print("Descargando LFW (puede tardar unos minutos)...")
    d = fetch_lfw_people(min_faces_per_person=per_person, color=True,
                         slice_=None, resize=1.0)
    count = {}
    for im, t in zip(d.images, d.target):
        name = d.target_names[t].replace(" ", "_")
        if name not in count and len(count) >= max_people:
            continue
        if count.get(name, 0) >= per_person:
            continue
        if im.max() <= 1.0:
            im = im * 255
        bgr = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
        p = out_dir / name
        p.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p / f"{count.get(name, 0):03d}.jpg"), bgr)
        count[name] = count.get(name, 0) + 1
    print(f"LFW: {len(count)} personas guardadas en {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="dataset")
    ap.add_argument("--lfw", action="store_true")
    ap.add_argument("--lfw-people", type=int, default=40)
    ap.add_argument("--per-person", type=int, default=10)
    ap.add_argument("--calib-n", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.images)
    if args.lfw:
        download_lfw(root / "lfw", args.lfw_people, args.per_person)

    app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(320, 320))

    crops_dir = WORK / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    rgb_crops, skipped = [], 0

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No hay imagenes en {root}/. Usa --lfw o agrega fotos.")

    for i, path in enumerate(files):
        img = cv2.imread(str(path))
        faces = app.get(img) if img is not None else []
        if not faces:
            skipped += 1
            continue
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        aligned = face_align.norm_crop(img, landmark=f.kps, image_size=112)  # BGR
        label = path.parent.name
        (crops_dir / label).mkdir(exist_ok=True)
        cv2.imwrite(str(crops_dir / label / f"{path.stem}.png"), aligned)
        rgb_crops.append(aligned[:, :, ::-1])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} imagenes...")

    print(f"Recortes validos: {len(rgb_crops)} | sin rostro: {skipped}")
    random.Random(0).shuffle(rgb_crops)
    calib = np.stack(rgb_crops[: args.calib_n]).astype(np.float32) / 255.0  # NHWC, 0-1
    np.save(WORK / "calib.npy", calib)
    print(f"Calibracion: {WORK / 'calib.npy'} {calib.shape}")


if __name__ == "__main__":
    main()
