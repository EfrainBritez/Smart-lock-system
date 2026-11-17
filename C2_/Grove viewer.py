#!/usr/bin/env python3
"""
C2 - Visor de diagnostico para el firmware tflm_fd_fm del Grove Vision AI V2.

El ejemplo emite por serial lineas JSON {"type": 1, "name": "INVOKE", "data": {...}}
con la imagen JPEG en base64 (campo "image"), las cajas de rostro
(fm_face_boxes: [x, y, w, h, score, target]) y la malla facial
(fm_points: 468 puntos [x, y, score, target] por rostro + 10 de iris).
Este script las decodifica y te muestra que esta viendo la camara.

Uso:
  python grove_viewer.py COM7                         # en vivo
  python grove_viewer.py --log putty.log --no-gui --save-dir frames   # desde un log de PuTTY
Teclas (con ventana): S = guardar frame, Q = salir
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_event(line):
    """Devuelve el dict 'data' de un evento INVOKE, o None si la linea no lo es."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None
    if msg.get("name") != "INVOKE":
        return None
    return msg.get("data", {})


def summarize(data):
    """-> (cajas con score>0, algo_tick, imagen BGR | None)"""
    boxes = [b for b in data.get("fm_face_boxes", []) if len(b) >= 6 and b[4] > 0]
    try:
        tick = data["algo_tick"][0][0]
    except (KeyError, IndexError, TypeError):
        tick = None
    img = None
    if data.get("image"):
        buf = np.frombuffer(base64.b64decode(data["image"]), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return boxes, tick, img


def lines_from_serial(port, baud):
    import serial  # pip install pyserial
    ser = serial.Serial(port, baud, timeout=2)
    while True:
        raw = ser.readline()
        if raw:
            yield raw.decode(errors="ignore")


def lines_from_file(path):
    with open(path, errors="ignore") as f:
        # PuTTY guarda los saltos como \r: los separamos tambien
        for chunk in f.read().replace("\r", "\n").split("\n"):
            yield chunk


def main():
    ap = argparse.ArgumentParser(description="Visor del firmware tflm_fd_fm")
    ap.add_argument("port", nargs="?", help="puerto serial (p. ej. COM7)")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--log", help="leer de un archivo de log en vez del puerto")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--save-dir", help="carpeta donde guardar frames")
    ap.add_argument("--save-every", type=int, default=0, help="guardar 1 de cada N frames")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    if not args.port and not args.log:
        sys.exit("Indica un puerto (COM7) o --log archivo")

    src = lines_from_file(args.log) if args.log else lines_from_serial(args.port, args.baud)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for line in src:
        data = parse_event(line)
        if data is None:
            continue
        boxes, tick, img = summarize(data)
        n += 1
        info = f"frame {n}: rostros={len(boxes)} algo_tick={tick}"
        if img is not None:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            info += f" imagen={img.shape[1]}x{img.shape[0]} brillo={g.mean():.0f} contraste={g.std():.1f}"
            if g.std() < 8:
                info += "  <-- imagen casi uniforme (lente tapada, muy oscuro o sin enfocar?)"
        else:
            info += " (sin imagen decodificable)"
        print(info)
        for b in boxes:
            print(f"    caja [x, y, w, h, score, target] = {b}")

        if img is not None and save_dir and args.save_every and n % args.save_every == 0:
            cv2.imwrite(str(save_dir / f"frame_{n:05d}.jpg"), img)

        if not args.no_gui and img is not None:
            view = img.copy()
            cv2.putText(view, f"rostros: {len(boxes)}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            try:
                cv2.imshow("Grove tflm_fd_fm", view)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                print("Esta instalacion de OpenCV no tiene ventanas; usa --no-gui")
                args.no_gui = True
                key = 255
            if key == ord("q"):
                break
            if key == ord("s") and save_dir:
                cv2.imwrite(str(save_dir / f"frame_{n:05d}.jpg"), img)
                print("    guardado")
        if args.max_frames and n >= args.max_frames:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()