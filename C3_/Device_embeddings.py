#!/usr/bin/env python3
"""
C3 - Herramienta para los embeddings que calcula el Grove (firmware C3a).

El firmware emite por serial el evento INVOKE con un campo "embedding":
  {"enabled", "valid", "scale_e6", "scale_bits", "zp", "crop": [x, y, lado], "tick", "data": [512 int8]}
junto con la imagen JPEG del mismo frame. Subcomandos:

  watch     Compara cada vector del modulo con el que calcula Python (modelo float32
            ONNX) sobre el MISMO recorte -> valida que el modelo corre bien en la NPU.
  collect   Guarda N vectores validos de una persona en device_db/<nombre>.npz.
  evaluate  Con las personas guardadas, mide misma persona vs distinta (verificacion).

Ejemplos:
  python device_embeddings.py watch COM7 --onnx ..\\C1_\\models\\embedding_fp32.onnx
  python device_embeddings.py collect COM7 Ana --count 30
  python device_embeddings.py evaluate
  python device_embeddings.py watch --log putty.log --onnx models/embedding_fp32.onnx
"""
import argparse
import base64
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ lectura de eventos
def parse_event(line):
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None
    return msg.get("data", {}) if msg.get("name") == "INVOKE" else None


def lines(args):
    if args.log:
        with open(args.log, errors="ignore") as f:
            yield from f.read().replace("\r", "\n").split("\n")
    else:
        import serial  # pip install pyserial
        ser = serial.Serial(args.port, args.baud, timeout=2)
        while True:
            raw = ser.readline()
            if raw:
                yield raw.decode(errors="ignore")


def decode_image(data):
    if not data.get("image"):
        return None
    buf = np.frombuffer(base64.b64decode(data["image"]), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def device_vector(emb):
    """Vector normalizado a partir del int8 del modulo, o None si no es valido."""
    if not emb or not emb.get("enabled") or not emb.get("valid") or len(emb.get("data", [])) != 512:
        return None
    scale = struct.unpack("<f", struct.pack("<I", int(emb["scale_bits"])))[0]
    v = (np.array(emb["data"], dtype=np.float32) - emb["zp"]) * scale
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


# ------------------------------------------------------------------ referencia en Python
class Reference:
    """Replica el preprocesado del firmware: recorte cuadrado del plano Y,
    reduccion bilineal a 112x112 (esquinas alineadas), Y -> 3 canales, ArcFace float32."""

    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, img_bgr, crop):
        x0, y0, side = crop
        h, w = img_bgr.shape[:2]
        if side <= 0 or x0 < 0 or y0 < 0 or x0 + side > w or y0 + side > h:
            return None
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[y0:y0 + side, x0:x0 + side]
        xs = np.linspace(0, side - 1, 112, dtype=np.float32)
        mx, my = np.meshgrid(xs, xs)
        r = cv2.remap(gray, mx, my, cv2.INTER_LINEAR)
        rgb = np.repeat(r[:, :, None], 3, axis=2)
        x = ((rgb.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        e = self.sess.run(None, {self.name: x})[0][0]
        return e / np.linalg.norm(e)


# ------------------------------------------------------------------ subcomandos
def cmd_watch(args):
    ref = Reference(args.onnx) if args.onnx else None
    cos, ticks, n = [], [], 0
    for line in lines(args):
        data = parse_event(line)
        if data is None:
            continue
        emb = data.get("embedding")
        n += 1
        if emb is None:
            print(f"frame {n}: el firmware NO trae campo 'embedding' (no es el firmware C3a)")
            continue
        if not emb.get("enabled"):
            print(f"frame {n}: embeddings DESACTIVADOS (modelo no grabado en flash o no cargo; "
                  "revisa el log de arranque 'EMB:')")
            continue
        v = device_vector(emb)
        if v is None:
            print(f"frame {n}: sin rostro (valid=0)")
            continue
        msg = f"frame {n}: crop={emb['crop']} tick={emb['tick']} ciclos"
        if ref is not None:
            img = decode_image(data)
            e = ref(img, emb["crop"]) if img is not None else None
            if e is not None:
                c = float(v @ e)
                cos.append(c)
                msg += f"  cos(modulo, python float32)={c:.3f}"
        ticks.append(emb["tick"])
        print(msg)
        if args.max_frames and len(ticks) >= args.max_frames:
            break
    if cos:
        print(f"\nResumen: {len(cos)} vectores | cos medio={np.mean(cos):.3f} minimo={np.min(cos):.3f}")
        print("Referencia: ~0.96 (int8 vs float, misma imagen) en C1; aqui se suman JPEG, "
              "reduccion y gris, asi que algo menos es normal. Valores muy bajos (<0.8) indican un problema.")
    if ticks:
        print(f"Tick medio del embedding: {np.mean(ticks):.0f} ciclos "
              f"(~{np.mean(ticks) / 400e6 * 1000:.0f} ms si el reloj es 400 MHz; suposicion)")


def cmd_collect(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / args.name).mkdir(exist_ok=True)
    vecs, crops = [], []
    for line in lines(args):
        data = parse_event(line)
        if data is None:
            continue
        v = device_vector(data.get("embedding"))
        if v is None:
            continue
        vecs.append(v)
        crops.append(data["embedding"]["crop"])
        img = decode_image(data)
        if img is not None:
            cv2.imwrite(str(out / args.name / f"frame_{len(vecs):04d}.jpg"), img)
        print(f"[{len(vecs)}/{args.count}] crop={crops[-1]}")
        if len(vecs) >= args.count:
            break
    if vecs:
        np.savez(out / f"{args.name}.npz", vectors=np.array(vecs), crops=np.array(crops))
        print(f"Guardado {out / (args.name + '.npz')} ({len(vecs)} vectores)")


def cmd_evaluate(args):
    files = sorted(Path(args.db).glob("*.npz"))
    people = {f.stem: np.load(f)["vectors"] for f in files}
    if len(people) < 2 or any(len(v) < 2 for v in people.values()):
        sys.exit("Se necesitan al menos 2 personas con 2+ vectores (usa 'collect').")
    gen, imp = [], []
    names = list(people)
    for n in names:
        v = people[n]
        s = v @ v.T
        gen += list(s[np.triu_indices(len(v), 1)])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            imp += list((people[a] @ people[b].T).ravel())
    gen, imp = np.array(gen), np.array(imp)
    ths = np.linspace(-0.2, 1.0, 601)
    acc = [((gen >= t).mean() + (imp < t).mean()) / 2 for t in ths]
    i = int(np.argmax(acc))
    print(f"Personas: {names} | vectores: {[len(people[n]) for n in names]}")
    print(f"Pares misma persona: {len(gen)} (media {gen.mean():.3f}) | distintas: {len(imp)} (media {imp.mean():.3f})")
    print(f"Mejor umbral {ths[i]:.3f} -> exactitud {acc[i]:.3f}")
    print(f"TPR @ FAR=1%: {(gen > np.quantile(imp, 0.99)).mean():.3f} (umbral {np.quantile(imp, 0.99):.3f})")
    print("Compara con la tabla de la etapa A / 05_evaluate.py para ver cuanto cuesta el recorte sin alineacion.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Embeddings del Grove (C3a)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("watch", "collect"):
        p = sub.add_parser(name)
        p.add_argument("port", nargs="?", help="puerto serial (COM7)")
        p.add_argument("--baud", type=int, default=921600)
        p.add_argument("--log", help="leer de un log en vez del puerto")
        if name == "watch":
            p.add_argument("--onnx", help="models/embedding_fp32.onnx (referencia float32)")
            p.add_argument("--max-frames", type=int, default=0)
            p.set_defaults(func=cmd_watch)
        else:
            p.add_argument("name")
            p.add_argument("--count", type=int, default=30)
            p.add_argument("--out", default="device_db")
            p.set_defaults(func=cmd_collect)
    e = sub.add_parser("evaluate")
    e.add_argument("--db", default="device_db")
    e.set_defaults(func=cmd_evaluate)
    a = ap.parse_args()
    if a.cmd in ("watch", "collect") and not a.port and not a.log:
        sys.exit("Indica un puerto (COM7) o --log archivo")
    a.func(a)