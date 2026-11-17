#!/usr/bin/env python3
"""
C2/C3 - Presupuesto de SRAM del Grove Vision AI V2 (ESTIMACION, no medicion).

Replica como el ejemplo tflm_fd_fm reparte el pool de SRAM con mm_reserve():
  pool = 2 MiB - (BOOT2NDLOADER_BASE - 0x34000000), con BOOT2NDLOADER_BASE
         = 0x3401F000 segun un comentario del SDK (no lo verifique en el binario).

MODO ACTIVO (verificado en el codigo): con UART_SEND_ALOGO_RESEULT el firmware inicia
el sensor en APP_DP_RES_YUV640x480_INP_SUBSAMPLE_2X -> el algoritmo recibe el plano Y
(GRIS, 1 canal) de 320x240. Por eso el JPEG que ves mide 320x240.
(Una version anterior de este script asumia RGB 320x240: era incorrecta.)

Uso: python memory_budget.py [--arena-kib 1210] [--mode uart|spi]
"""
import argparse

KIB = 1024
POOL = 2 * 1024 * KIB - 0x1F000

SENSOR = {  # (wdma1+wdma2 compartido, wdma3 raw, jpeg autofill)
    "uart": (19200, 115200, 100),   # YUV 320x240 (subsample 2X)
    "spi": (76800, 460800, 100),    # YUV 640x480 (subsample 1X)
}


def report(title, items):
    total = sum(items.values())
    print(f"\n{title}")
    for k, v in items.items():
        print(f"  {k:<34}{v / KIB:8.1f} KiB")
    print(f"  {'TOTAL':<34}{total / KIB:8.1f} KiB   libre: {(POOL - total) / KIB:6.1f} KiB"
          f"  ({'OK' if total <= POOL else 'NO CABE'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena-kib", type=float, default=1210.0,
                    help="arena TFLM del diseno nuevo (Vela pide 1176 KiB + holgura)")
    ap.add_argument("--mode", choices=SENSOR, default="uart")
    a = ap.parse_args()

    w, h = (320, 240) if a.mode == "uart" else (640, 480)
    s1, s3, sj = SENSOR[a.mode]
    items = {
        "sensor wdma1 (JPEG)": s1,
        "sensor wdma3 (raw YUV)": s3,
        "jpeg autofill": sj,
        "resized_img (192x192x1)": 192 * 192,
        f"crop_img ({w}x{h}x1)": w * h,
        f"pad_img ({w}x{w}x1)": w * w,
        "crop ojos (2x64x64)": 2 * 4096,
    }
    print(f"Modo: {a.mode} ({w}x{h} gris) | pool de SRAM: {POOL / KIB:.0f} KiB")
    report("Ejemplo tflm_fd_fm actual (arena 460 KiB)", {**items, "tensor arena": 460 * KIB})
    report(f"C3a: arena {a.arena_kib:.0f} KiB (mismos buferes)",
           {**items, "tensor arena": int(a.arena_kib * KIB)})


if __name__ == "__main__":
    main()