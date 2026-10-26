# Embedded Face Recognition

Face recognition for registered people using a **Seeed Wio Terminal** and a
**Grove Vision AI Module V2** (Himax WiseEye2, Cortex-M55 + Ethos-U55).

## Stages

- **Stage A (done):** the Grove module streams frames to the PC, which
  extracts face embeddings with InsightFace and compares them against a local
  database. The result is sent to the Wio Terminal, which shows it on screen
  and beeps.
- **Stage C (in progress):** run the embedding model (quantized MobileFaceNet)
  on the Grove's NPU, and move the database and matching to the Wio Terminal
  so the PC is no longer needed.
  Plan por sub-etapas
- **Sub-stages**
  
  C1, solo PC: exportar el modelo, cuantizarlo a int8, pasarlo por Vela y evaluarlo en Python. No necesita hardware y ya nos dice si el modelo es viable.


  C2, toolchain: compilar y cargar el ejemplo tflm_fd_fm sin modificarlo, para confirmar que el entorno de compilación y la carga funcionan.


  C3, embedding en el Grove: integrar el modelo, hacer que el módulo envíe el vector por serial a la PC y compararlo con la referencia de C1.


  C4, Wio autónomo: conectar Grove y Wio directamente por UART, mover la base de datos y la comparación al Wio, y retirar la PC.


  C5, pulido: umbral, rendimiento y, si el sistema controla un acceso, detección de fotos falsas.

## Structure

- `face_system.py`: register, list, delete and recognize people (Python)
- `wio_terminal/wio_terminal.ino`: display and buzzer receiver (Arduino)

## Quick start

    pip install -r requirements.txt
    python face_system.py register Ana --samples 5
    python face_system.py --source grove --grove-port COM5 recognize --wio-port COM7

## License

The source code in this repository is released under the MIT License.

This project uses pretrained models from [InsightFace](https://github.com/deepinsight/insightface)
(e.g. `buffalo_l`), which are **not included** in this repo. These models are
available for **non-commercial research purposes only**. Any commercial use
requires a separate license from InsightFace.