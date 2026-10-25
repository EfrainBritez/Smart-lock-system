#!/usr/bin/env python3
"""
Etapa A: reconocimiento facial en PC (InsightFace) con fuente de video
intercambiable (webcam o Grove Vision AI V2) y salida opcional al Wio Terminal.

Uso:
    python face_system.py register Ana --samples 5
    python face_system.py recognize --threshold 0.40
    python face_system.py list
    python face_system.py delete Ana

    # Con el Grove Vision AI V2 como camara (por USB serial):
    python face_system.py --source grove --grove-port COM5 recognize

    # Enviando el resultado al Wio Terminal:
    python face_system.py recognize --wio-port COM7
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

DB_PATH = Path(__file__).with_name("faces_db.json")


# ---------------------------------------------------------------------------
# Fuentes de video (misma interfaz: read() -> frame BGR o None, close())
# ---------------------------------------------------------------------------
class WebcamSource:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la webcam {index}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class GroveSource:
    """
    Pide un frame al Grove Vision AI V2 con el protocolo AT de SSCMA.
    AT+INVOKE=1,0,0 -> el modulo responde una linea JSON con el campo
    data.image en base64 (JPEG). Verifica el baudrate y el formato exacto
    con tu version de firmware.
    """

    def __init__(self, port, baud=921600):
        import serial  # pip install pyserial

        self.ser = serial.Serial(port, baud, timeout=2)

    def read(self):
        self.ser.reset_input_buffer()
        self.ser.write(b"AT+INVOKE=1,0,0\r")
        deadline = time.time() + 3
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="ignore").strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("name") == "INVOKE":
                b64 = msg.get("data", {}).get("image")
                if b64:
                    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
                    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return None

    def close(self):
        self.ser.close()


def open_source(args):
    if args.source == "grove":
        if not args.grove_port:
            sys.exit("--grove-port es obligatorio con --source grove")
        return GroveSource(args.grove_port)
    return WebcamSource(args.camera)


# ---------------------------------------------------------------------------
# Base de datos de rostros: {nombre: [embedding, ...]}
# ---------------------------------------------------------------------------
def load_db():
    if not DB_PATH.exists():
        return {}
    raw = json.loads(DB_PATH.read_text())
    return {n: np.array(e, dtype=np.float32) for n, e in raw.items()}


def save_db(db):
    DB_PATH.write_text(json.dumps({n: e.tolist() for n, e in db.items()}))


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def load_model(model_name="buffalo_l"):
    # buffalo_s es mas liviano si tu PC va lenta
    app = FaceAnalysis(
        name=model_name,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def identify(embedding, db, threshold):
    """Similitud coseno (los embeddings ya vienen normalizados)."""
    best_name, best_score = None, -1.0
    for name, embs in db.items():
        score = float(np.max(embs @ embedding))
        if score > best_score:
            best_name, best_score = name, score
    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


# ---------------------------------------------------------------------------
# Comunicacion con el Wio Terminal: lineas "OK,nombre,score" / "UNKNOWN,,score"
# ---------------------------------------------------------------------------
class WioLink:
    def __init__(self, port, baud=115200):
        import serial

        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.last = None

    def send(self, name, score):
        msg = f"OK,{name},{score:.2f}\n" if name else f"UNKNOWN,,{score:.2f}\n"
        if msg != self.last:  # solo cuando cambia el estado
            self.ser.write(msg.encode())
            self.last = msg

    def send_none(self):
        if self.last != "NONE\n":
            self.ser.write(b"NONE\n")
            self.last = "NONE\n"


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------
def cmd_register(args):
    app = load_model(args.model)
    db = load_db()
    src = open_source(args)
    samples = []
    print(f"Registrando a '{args.name}'. ESPACIO = capturar, Q = terminar.")
    print("Varia el angulo y la iluminacion entre capturas.")

    try:
        while len(samples) < args.samples:
            frame = src.read()
            if frame is None:
                continue
            faces = app.get(frame)
            view = frame.copy()
            for f in faces:
                x1, y1, x2, y2 = f.bbox.astype(int)
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(view, f"{len(samples)}/{args.samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Registro", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if len(faces) != 1:
                    print(f"Se necesita exactamente 1 rostro (hay {len(faces)}).")
                    continue
                samples.append(faces[0].normed_embedding)
                print(f"Muestra {len(samples)} capturada.")
    finally:
        src.close()
        cv2.destroyAllWindows()

    if not samples:
        print("No se registro ninguna muestra.")
        return
    new = np.stack(samples).astype(np.float32)
    db[args.name] = np.vstack([db[args.name], new]) if args.name in db else new
    save_db(db)
    print(f"'{args.name}' guardado con {len(db[args.name])} embeddings.")


def cmd_recognize(args):
    db = load_db()
    if not db:
        sys.exit("La base de datos esta vacia. Registra a alguien primero.")
    app = load_model(args.model)
    src = open_source(args)
    wio = WioLink(args.wio_port) if args.wio_port else None

    streak_name, streak = None, 0  # confirmacion por frames consecutivos
    print("Reconociendo. Presiona Q para salir.")

    try:
        while True:
            frame = src.read()
            if frame is None:
                continue
            faces = app.get(frame)
            label, color = "Sin rostro", (128, 128, 128)

            if faces:
                face = largest_face(faces)
                name, score = identify(face.normed_embedding, db, args.threshold)
                streak = streak + 1 if name == streak_name else 1
                streak_name = name

                confirmed = streak >= args.confirm
                if name:
                    label, color = f"{name} ({score:.2f})", (0, 200, 0)
                else:
                    label, color = f"Desconocido ({score:.2f})", (0, 0, 255)
                if confirmed and wio:
                    wio.send(name, score)

                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                streak_name, streak = None, 0
                if wio:
                    wio.send_none()

            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("Reconocimiento", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        src.close()
        cv2.destroyAllWindows()


def cmd_list(_args):
    db = load_db()
    if not db:
        print("Base de datos vacia.")
    for name, embs in db.items():
        print(f"{name}: {len(embs)} embeddings")


def cmd_delete(args):
    db = load_db()
    if args.name in db:
        del db[args.name]
        save_db(db)
        print(f"'{args.name}' eliminado.")
    else:
        print(f"'{args.name}' no existe.")


def main():
    p = argparse.ArgumentParser(description="Reconocimiento facial - etapa A")
    p.add_argument("--source", choices=["webcam", "grove"], default="webcam")
    p.add_argument("--camera", type=int, default=0, help="indice de webcam")
    p.add_argument("--grove-port", help="puerto serial del Grove Vision AI V2")
    p.add_argument("--model", default="buffalo_l", help="buffalo_l o buffalo_s")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="registrar una persona")
    r.add_argument("name")
    r.add_argument("--samples", type=int, default=5)
    r.set_defaults(func=cmd_register)

    g = sub.add_parser("recognize", help="reconocer en tiempo real")
    g.add_argument("--threshold", type=float, default=0.40)
    g.add_argument("--confirm", type=int, default=3,
                   help="frames consecutivos para confirmar identidad")
    g.add_argument("--wio-port", help="puerto serial del Wio Terminal")
    g.set_defaults(func=cmd_recognize)

    sub.add_parser("list", help="listar registrados").set_defaults(func=cmd_list)

    d = sub.add_parser("delete", help="eliminar una persona")
    d.add_argument("name")
    d.set_defaults(func=cmd_delete)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()