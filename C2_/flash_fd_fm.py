#!/usr/bin/env python3
"""
C2 - Graba el firmware y los modelos en el Grove Vision AI V2 usando el
xmodem_send.py del SDK de Himax. Funciona en Windows, Linux y macOS.

Las direcciones de flash de cada modelo se leen del nombre del archivo
(p. ej. 0_fd_0x200000.tflite -> 0x200000), como hace el SDK.

Uso:
  python flash_fd_fm.py COM5 --sdk C:\\ruta\\Seeed_Grove_Vision_AI_Module_V2
  python flash_fd_fm.py COM5 --sdk ... --dry-run     # solo muestra el comando
  python flash_fd_fm.py COM5 --sdk ... --restore     # firmware de fabrica (SenseCraft)

Cuando termine pregunta si quieres reiniciar: pulsa el boton RESET del modulo.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


def build_cmd(sdk, port, app, restore, image):
    cmd = [sys.executable, "xmodem/xmodem_send.py", f"--port={port}",
           "--baudrate=921600", "--protocol=xmodem"]
    if restore:
        imgs = sorted(sdk.glob("Seeed_SenseCraft_AI*.img"))
        if not imgs:
            sys.exit("No encuentro Seeed_SenseCraft_AI*.img en el SDK.")
        cmd.append(f"--file={imgs[-1].name}")
        return cmd

    img = Path(image) if image else \
        Path("we2_image_gen_local/output_case1_sec_wlcsp/output.img")
    if not (sdk / img).exists() and not img.is_absolute():
        sys.exit(f"No existe {sdk / img}. Compila primero (build_fd_fm.sh) "
                 "o copia output.img ahi / usa --image.")
    cmd.append(f"--file={img}")
    models = sorted((sdk / "model_zoo" / app).glob("*.tflite"))
    if not models:
        sys.exit(f"No hay modelos en model_zoo/{app}")
    for m in models:
        mm = re.search(r"_(0x[0-9A-Fa-f]+)\.tflite$", m.name)
        if not mm:
            print(f"Aviso: {m.name} no trae direccion en el nombre, lo omito.")
            continue
        rel = m.relative_to(sdk).as_posix()
        cmd.append(f"--model={rel} {mm.group(1)} 0x00000")
    return cmd


def main():
    ap = argparse.ArgumentParser(description="Graba firmware y modelos (Himax WE2)")
    ap.add_argument("port", help="puerto serial del modulo, p. ej. COM5 o /dev/ttyACM0")
    ap.add_argument("--sdk", required=True, help="carpeta Seeed_Grove_Vision_AI_Module_V2")
    ap.add_argument("--app", default="tflm_fd_fm", help="carpeta de model_zoo")
    ap.add_argument("--image", help="output.img alternativo")
    ap.add_argument("--restore", action="store_true", help="restaurar firmware de fabrica")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sdk = Path(args.sdk).resolve()
    if not (sdk / "xmodem" / "xmodem_send.py").exists():
        sys.exit(f"{sdk} no parece el SDK de Himax (falta xmodem/xmodem_send.py)")

    cmd = build_cmd(sdk, args.port, args.app, args.restore, args.image)
    print("Carpeta:", sdk)
    print("Comando:", " ".join(f'"{c}"' if " " in c else c for c in cmd), "\n")
    if args.dry_run:
        return
    print("Cierra Tera Term / minicom / Monitor Serie antes de seguir.")
    sys.exit(subprocess.run(cmd, cwd=sdk).returncode)


if __name__ == "__main__":
    main()
