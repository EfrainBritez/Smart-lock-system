#!/usr/bin/env python3
"""
Paso 4: compilar el modelo int8 con Vela para la NPU Ethos-U55 del Grove.

Genera work/vela/<modelo>_vela.tflite (el que se grabara en el modulo) y
muestra el resumen: cuantas operaciones van a la NPU vs. la CPU, la memoria
SRAM necesaria y una ESTIMACION de tiempo de inferencia.

Uso: python 04_vela.py [--tflite RUTA] [--vela-config RUTA]
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys

from common import ROOT, WORK

ap = argparse.ArgumentParser()
ap.add_argument("--tflite", default=None)
ap.add_argument("--vela-config", default=str(ROOT / "vela_config.ini"))
args = ap.parse_args()

if not shutil.which("vela"):
    sys.exit("No encuentro 'vela'. Instala con: pip install ethos-u-vela")

src = args.tflite or next((WORK / "tflite").glob("*_full_integer_quant.tflite"))
out_dir = WORK / "vela"
cmd = [
    "vela", str(src),
    "--accelerator-config", "ethos-u55-64",
    "--config", args.vela_config,
    "--system-config", "My_Sys_Cfg",
    "--memory-mode", "My_Mem_Mode_Parent",
    "--optimise", "Performance",
    "--verbose-performance",
    "--output-dir", str(out_dir),
]
print(" ".join(cmd), "\n")
res = subprocess.run(cmd, capture_output=True, text=True)
log = res.stdout + res.stderr
(out_dir.parent / "vela_report.txt").write_text(log)
if res.returncode != 0:
    print(log)
    sys.exit("Vela fallo (ver mensaje arriba).")

keys = ("Accelerator configuration", "Total SRAM used", "Total Off-chip Flash used",
        "Total On-chip Flash used", "CPU operators", "NPU operators",
        "Neural network macs")
print("=== Resumen de Vela ===")
for line in log.splitlines():
    if any(k in line for k in keys):
        print(" ", re.sub(r"\s{2,}", "  ", line.strip()))

# Estimacion de tiempo (CSV de resumen de Vela)
try:
    row = next(csv.DictReader(open(next(out_dir.glob("*summary*.csv")), newline="")))
    t, ips = float(row["inference_time"]), float(row["inferences_per_second"])
    print(f"  Tiempo estimado: {t * 1000:.0f} ms por rostro ({ips:.1f} inferencias/s)")
    print("  (estimacion: depende de que vela_config.ini coincida con el hardware real)")
except (StopIteration, KeyError, ValueError):
    pass

cpu = re.search(r"CPU operators\s*=\s*(\d+)", log)
npu = re.search(r"NPU operators\s*=\s*(\d+)", log)
print()
if cpu and int(cpu.group(1)) == 0:
    print("OK: todas las operaciones van a la NPU.")
elif cpu:
    print(f"ATENCION: {cpu.group(1)} operaciones caen a la CPU (mas lento).")
    print("Revisa work/vela_report.txt y work/vela/*per-layer.csv para ver cuales.")
print(f"\nModelo para el Grove: {next(out_dir.glob('*_vela.tflite'))}")
