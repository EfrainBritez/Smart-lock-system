#!/usr/bin/env python3
"""
C2 - Presupuesto de SRAM del Grove Vision AI V2 (ESTIMACION, no medicion).

Replica como el ejemplo tflm_fd_fm reparte el pool de SRAM con mm_reserve():
  pool = 2 MiB - (BOOT2NDLOADER_BASE - 0x34000000), con BOOT2NDLOADER_BASE
         = 0x3401F000 segun un comentario del SDK (no lo verifique en el binario).
Los tamanos de buffers salen de cvapp_fd_fm.cpp y cisdp_sensor.c (camara
OV5647, modo RGB 320x240, que es la rama por defecto).

Uso: python memory_budget.py [--arena-kib 1176] [--keep-pad]
"""
import argparse

KIB = 1024
POOL = 2 * 1024 * KIB - 0x1F000


def report(title, items):
    total = sum(items.values())
    print(f"\n{title}")
    for k, v in items.items():
        print(f"  {k:<28}{v / KIB:8.0f} KiB")
    print(f"  {'TOTAL':<28}{total / KIB:8.0f} KiB   libre: {(POOL - total) / KIB:5.0f} KiB"
          f"  ({'OK' if total <= POOL else 'NO CABE'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena-kib", type=float, default=1176.0,
                    help="'Total SRAM used' del reporte de Vela")
    ap.add_argument("--keep-pad", action="store_true",
                    help="conservar pad_img (320x320x3) en nuestro diseno")
    a = ap.parse_args()

    sensor = {"sensor wdma1 + wdma3 (320x240)": 19200 + 230400}
    common = {"resized (192x192x3)": 192 * 192 * 3, "crop_img (320x240x3)": 320 * 240 * 3}
    pad = {"pad_img (320x320x3)": 320 * 320 * 3}

    print(f"Pool de SRAM disponible: {POOL / KIB:.0f} KiB")
    report("Ejemplo tflm_fd_fm actual", {**sensor, "tensor arena": 460 * KIB, **common, **pad,
                                         "crop ojos (2x64x64)": 2 * 4096})
    ours = {**sensor, "tensor arena (embedding)": int(a.arena_kib * KIB) + 1224, **common,
            "rostro alineado (112x112x3)": 112 * 112 * 3}
    if a.keep_pad:
        ours.update(pad)
    report("Nuestro diseno (detector + embedding)" + (" con pad_img" if a.keep_pad else " sin pad_img"), ours)


if __name__ == "__main__":
    main()
