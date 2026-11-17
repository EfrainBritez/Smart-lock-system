#!/usr/bin/env python3
"""
Paso 6 (C3): cuanto pierde el modelo si recibe la imagen en ESCALA DE GRISES.

En el modo UART, el firmware tflm_fd_fm entrega al algoritmo solo el plano Y
(gris) de 320x240. ArcFace se entreno con imagenes a color, asi que hay que medir
el efecto de alimentarlo con Y replicado en los 3 canales (YYY).

Compara, sobre los recortes de work/crops/<persona>/*.png:
  - color vs gris con el modelo float32 (ONNX)
  - color vs gris con el modelo int8 (TFLite, antes de Vela), si existe
y reporta verificacion (misma persona vs distinta) y la similitud entre el
embedding a color y el gris de la MISMA imagen.

Uso: python 06_gray_vs_color.py [--max-images 600]
"""
import argparse
import importlib.util
import random
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

from common import FP32_ONNX, WORK, l2norm, normalize_rgb

spec = importlib.util.spec_from_file_location("ev05", str(__import__("pathlib").Path(__file__).with_name("05_evaluate.py")))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def to_gray_yyy(bgr):
    """Replica lo que hara el firmware: plano Y (BT.601) copiado a 3 canales."""
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return np.repeat(y[:, :, None], 3, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=600)
    ap.add_argument("--pairs", type=int, default=3000)
    ap.add_argument("--no-int8", action="store_true",
                    help="solo modelo float32 (no requiere TensorFlow ni ai_edge_litert)")
    args = ap.parse_args()

    files = sorted((WORK / "crops").rglob("*.png"))
    random.Random(0).shuffle(files)
    files = files[: args.max_images]
    labels = [f.parent.name for f in files]

    sess = ort.InferenceSession(str(FP32_ONNX), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    tfl = next((WORK / "tflite").glob("*_full_integer_quant.tflite"), None)
    q = None
    if tfl and not args.no_int8:
        try:
            q = ev.TfliteEmbedder(tfl)
        except ImportError:
            print("Aviso: este Python no tiene interprete TFLite (ai_edge_litert / tensorflow).\n"
                  "Comparo solo float32, que basta para decidir gris vs color. Para incluir int8,\n"
                  "ejecuta este script en el entorno donde corriste 05_evaluate.py (p. ej. WSL).\n")

    E = defaultdict(list)
    for f in files:
        bgr = cv2.imread(str(f))
        for tag, img in (("color", bgr), ("gris", to_gray_yyy(bgr))):
            rgb = img[:, :, ::-1]
            x = normalize_rgb(rgb).transpose(2, 0, 1)[None]
            E[f"{tag} float32"].append(sess.run(None, {name: x})[0][0])
            if q:
                E[f"{tag} int8"].append(q.embed(rgb))
    E = {k: l2norm(np.array(v)) for k, v in E.items()}

    by_person = defaultdict(list)
    for i, l in enumerate(labels):
        by_person[l].append(i)
    rng = random.Random(1)
    gen = [(a, b) for idx in by_person.values() for a in idx for b in idx if a < b]
    rng.shuffle(gen)
    gen = gen[: args.pairs]
    people = list(by_person)
    imp = []
    while len(people) > 1 and len(imp) < len(gen):
        p1, p2 = rng.sample(people, 2)
        imp.append((rng.choice(by_person[p1]), rng.choice(by_person[p2])))
    if not gen or not imp:
        raise SystemExit("Se necesitan al menos 2 personas con 2+ fotos cada una.")

    print(f"Imagenes: {len(files)} | pares misma persona: {len(gen)} | pares distintas: {len(imp)}\n")
    print(f"{'variante':<16}{'mejor umbral':>13}{'exactitud':>10}{'TPR@FAR=1%':>12}"
          f"{'sim. misma':>12}{'sim. distinta':>14}")
    for k, em in E.items():
        g = np.array([em[a] @ em[b] for a, b in gen])
        im = np.array([em[a] @ em[b] for a, b in imp])
        th, acc, tpr = ev.verification_metrics(g, im)
        print(f"{k:<16}{th:13.3f}{acc:10.3f}{tpr:12.3f}{g.mean():12.3f}{im.mean():14.3f}")

    print("\nSimilitud coseno color vs gris de la MISMA imagen:")
    for kind in ("float32", "int8"):
        if f"color {kind}" in E:
            c = np.sum(E[f"color {kind}"] * E[f"gris {kind}"], axis=1)
            print(f"  {kind:8} media={c.mean():.3f}  minimo={c.min():.3f}")
    print("\nLectura: si 'gris' queda cerca de 'color' en exactitud y TPR, el firmware puede usar "
          "el plano Y tal cual. Si cae mucho, hay que obtener color (YUV->RGB) en el firmware.")


if __name__ == "__main__":
    main()