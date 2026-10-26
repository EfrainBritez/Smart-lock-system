#!/usr/bin/env python3
"""
Paso 5: evaluar el modelo int8 frente al original float32, en Python.

Mide dos cosas sobre los recortes de work/crops/<persona>/*.png:
  A) Similitud coseno entre el embedding float y el int8 de la MISMA imagen
     (cuanto dano hizo la cuantizacion).
  B) Verificacion de identidad con pares de la misma persona vs. distintas,
     para float e int8 (si el sistema sigue sirviendo en la practica).

Importante: se evalua el .tflite int8 *antes* de Vela. El archivo *_vela.tflite
contiene una operacion personalizada de la NPU Ethos-U que un interprete de PC
no sabe ejecutar. Vela esta pensado para reproducir los mismos resultados
enteros, y eso se confirma en la etapa C3 contra el hardware real.

Uso: python 05_evaluate.py [--tflite RUTA] [--max-images 600]
"""
import argparse
import random
from collections import Counter, defaultdict

import cv2
import numpy as np
import onnxruntime as ort

from common import FP32_ONNX, WORK, l2norm, normalize_rgb


def load_interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=str(path))


class TfliteEmbedder:
    def __init__(self, path):
        self.it = load_interpreter(path)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def describe(self):
        print(f"  entrada : {self.inp['shape']} {np.dtype(self.inp['dtype']).name} "
              f"scale={self.inp['quantization'][0]:.6f} zp={self.inp['quantization'][1]}")
        print(f"  salida  : {self.out['shape']} {np.dtype(self.out['dtype']).name} "
              f"scale={self.out['quantization'][0]:.6f} zp={self.out['quantization'][1]}")
        try:
            ops = Counter(d["op_name"] for d in self.it._get_ops_details())
            print("  operadores:", dict(ops))
        except Exception:
            pass

    def embed(self, rgb_u8):
        x = normalize_rgb(rgb_u8)[None]                       # NHWC float
        dt = np.dtype(self.inp["dtype"])
        if dt in (np.int8, np.uint8):
            scale, zp = self.inp["quantization"]
            info = np.iinfo(dt)
            x = np.clip(np.round(x / scale + zp), info.min, info.max).astype(dt)
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"]).astype(np.float32)
        if np.dtype(self.out["dtype"]) in (np.int8, np.uint8):
            scale, zp = self.out["quantization"]
            y = (y - zp) * scale
        return y[0]


def verification_metrics(gen, imp):
    ths = np.linspace(-0.2, 1.0, 601)
    acc = [((gen >= t).mean() + (imp < t).mean()) / 2 for t in ths]
    i = int(np.argmax(acc))
    tpr_at_far1 = float((gen > np.quantile(imp, 0.99)).mean())
    return ths[i], acc[i], tpr_at_far1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default=None)
    ap.add_argument("--max-images", type=int, default=600)
    ap.add_argument("--pairs", type=int, default=3000)
    args = ap.parse_args()

    tfl_path = args.tflite or next(
        (WORK / "tflite").glob("*_full_integer_quant.tflite"))
    print(f"Modelo int8: {tfl_path}")
    q = TfliteEmbedder(tfl_path)
    q.describe()

    sess = ort.InferenceSession(str(FP32_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    files = sorted((WORK / "crops").rglob("*.png"))
    random.Random(0).shuffle(files)
    files = files[: args.max_images]
    labels, fp, qq = [], [], []
    for f in files:
        rgb = cv2.imread(str(f))[:, :, ::-1]
        x = normalize_rgb(rgb).transpose(2, 0, 1)[None]        # NCHW para ONNX
        fp.append(sess.run(None, {in_name: x})[0][0])
        qq.append(q.embed(rgb))
        labels.append(f.parent.name)
    fp, qq = l2norm(np.array(fp)), l2norm(np.array(qq))
    np.savez(WORK / "reference_embeddings.npz", files=[str(f) for f in files],
             labels=labels, fp32=fp, int8=qq)

    # ---- A) Dano de la cuantizacion ---------------------------------------
    cos = np.sum(fp * qq, axis=1)
    print(f"\n[A] Similitud coseno float32 vs int8 (misma imagen), N={len(cos)}")
    print(f"    media={cos.mean():.4f}  minimo={cos.min():.4f}  "
          f"percentil 5={np.percentile(cos, 5):.4f}")
    print("    ->", "OK (>= 0.95)" if cos.mean() >= 0.95 else
          "BAJO: probar --per-tensor, mas calibracion, o revisar capas")

    # ---- B) Verificacion de identidad -------------------------------------
    by_person = defaultdict(list)
    for i, l in enumerate(labels):
        by_person[l].append(i)
    rng = random.Random(1)
    gen_pairs = [p for idx in by_person.values() for p in
                 [(a, b) for a in idx for b in idx if a < b]]
    rng.shuffle(gen_pairs)
    gen_pairs = gen_pairs[: args.pairs]
    people = list(by_person)
    imp_pairs = []
    while len(people) > 1 and len(imp_pairs) < len(gen_pairs):
        p1, p2 = rng.sample(people, 2)
        imp_pairs.append((rng.choice(by_person[p1]), rng.choice(by_person[p2])))

    if not gen_pairs or not imp_pairs:
        print("\n[B] Se necesitan al menos 2 personas con 2+ fotos cada una.")
        return
    print(f"\n[B] Verificacion: {len(gen_pairs)} pares misma persona, "
          f"{len(imp_pairs)} pares distintas")
    print(f"    {'modelo':8} {'mejor umbral':>13} {'exactitud':>10} "
          f"{'TPR@FAR=1%':>11} {'sim. misma':>11} {'sim. distinta':>14}")
    for name, E in (("float32", fp), ("int8", qq)):
        g = np.array([E[a] @ E[b] for a, b in gen_pairs])
        im = np.array([E[a] @ E[b] for a, b in imp_pairs])
        th, acc, tpr = verification_metrics(g, im)
        print(f"    {name:8} {th:13.3f} {acc:10.3f} {tpr:11.3f} "
              f"{g.mean():11.3f} {im.mean():14.3f}")
    print(f"\nEmbeddings de referencia guardados en {WORK / 'reference_embeddings.npz'}")


if __name__ == "__main__":
    main()
