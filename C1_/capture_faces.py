#!/usr/bin/env python3
"""
Captura rostros de 112x112 (ya alineados, como los espera el modelo) y los
guarda en <out>/<nombre>/  para usarlos con 02_prepare_data.py.

Uso:
  python capture_faces.py Ana                       # webcam, manual (ESPACIO)
  python capture_faces.py Ana --auto --count 30     # captura sola cada 0.7 s
  python capture_faces.py Ana --out dataset --camera 1
  python capture_faces.py Ana --source grove --grove-port COM5

Teclas:  ESPACIO = capturar   A = activar/desactivar modo automatico   Q = salir

Consejos: usa la MISMA camara con la que vas a reconocer, y varia el angulo,
la distancia y la iluminacion entre capturas.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

SIZE = 112


class WebcamSource:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la camara {index}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


def open_source(args):
    if args.source == "grove":
        if not args.grove_port:
            sys.exit("--grove-port es obligatorio con --source grove")
        # Reutiliza la clase de face_system.py (carpeta superior)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        try:
            from face_system import GroveSource
        except ImportError:
            sys.exit("No encuentro face_system.py en la carpeta superior.")
        return GroveSource(args.grove_port)
    return WebcamSource(args.camera)


def analyze(app, frame, min_face=60, min_score=0.6):
    """Devuelve (face, rostro_alineado_112x112 | None, mensaje)."""
    faces = app.get(frame)
    if not faces:
        return None, None, "Sin rostro"
    if len(faces) > 1:
        return None, None, f"{len(faces)} rostros: debe haber solo uno"
    f = faces[0]
    if f.det_score < min_score:
        return f, None, "Deteccion dudosa, mejora la luz"
    if f.bbox[2] - f.bbox[0] < min_face:
        return f, None, "Acercate a la camara"
    aligned = face_align.norm_crop(frame, landmark=f.kps, image_size=SIZE)
    return f, aligned, "OK"


def next_path(folder, name):
    n = len(list(folder.glob("*.png")))
    while (folder / f"{name}_{n:03d}.png").exists():
        n += 1
    return folder / f"{name}_{n:03d}.png"


def main():
    ap = argparse.ArgumentParser(description="Captura de rostros 112x112")
    ap.add_argument("name", help="nombre de la persona (nombre de la carpeta)")
    ap.add_argument("--out", default="dataset", help="carpeta raiz (default: dataset)")
    ap.add_argument("--count", type=int, default=20, help="fotos a tomar (0 = sin limite)")
    ap.add_argument("--auto", action="store_true", help="captura automatica")
    ap.add_argument("--interval", type=float, default=0.7, help="segundos entre capturas auto")
    ap.add_argument("--source", choices=["webcam", "grove"], default="webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--grove-port")
    ap.add_argument("--min-face", type=int, default=60, help="ancho minimo del rostro en px")
    args = ap.parse_args()

    name = args.name.strip().replace(" ", "_")
    folder = Path(args.out) / name
    folder.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    src = open_source(args)

    saved, auto, last = 0, args.auto, 0.0
    print(f"Guardando en: {folder.resolve()}")
    print("ESPACIO = capturar | A = auto on/off | Q = salir")

    try:
        while args.count == 0 or saved < args.count:
            frame = src.read()
            if frame is None:
                continue
            face, aligned, msg = analyze(app, frame, args.min_face)
            ok = aligned is not None

            view = frame.copy()
            if face is not None:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(view, (x1, y1), (x2, y2),
                              (0, 200, 0) if ok else (0, 0, 255), 2)
            if ok:  # vista previa de lo que se guardaria
                view[8:8 + SIZE, view.shape[1] - SIZE - 8: view.shape[1] - 8] = aligned
            total = f"/{args.count}" if args.count else ""
            cv2.putText(view, f"{name}: {saved}{total}  {'AUTO' if auto else 'MANUAL'}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(view, msg, (10, view.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0) if ok else (0, 0, 255), 2)
            cv2.imshow("Captura de rostros", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                auto = not auto
            want = (key == ord(" ")) or (auto and time.time() - last >= args.interval)
            if want:
                if ok:
                    path = next_path(folder, name)
                    cv2.imwrite(str(path), aligned)
                    saved += 1
                    last = time.time()
                    print(f"[{saved}] {path.name}")
                elif key == ord(" "):
                    print(f"No se guardo: {msg}")
    finally:
        src.close()
        cv2.destroyAllWindows()

    print(f"Listo: {saved} fotos nuevas en {folder} ({len(list(folder.glob('*.png')))} en total).")


if __name__ == "__main__":
    main()