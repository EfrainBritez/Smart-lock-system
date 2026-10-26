# C1 proposito

C1, solo PC: exportar el modelo, cuantizarlo a int8, pasarlo por Vela y evaluarlo en Python. No necesita hardware y ya nos dice si el modelo es viable.

# C1: modelo de embeddings para el Grove Vision AI V2 (solo PC)

Pipeline: ONNX float32 -> TFLite int8 -> Vela (NPU Ethos-U55) -> evaluacion.

    pip install -r requirements_c1.txt        # recomendado: Linux / WSL2, venv nuevo

    python 01_get_model.py                    # copia MobileFaceNet (buffalo_s) a models/
    python 02_prepare_data.py --lfw           # o: --images dataset/  (dataset/<persona>/*.jpg)
    python 03_quantize.py                     # work/tflite/*_full_integer_quant.tflite
    python 04_vela.py                         # work/vela/*_vela.tflite + resumen
    python 05_evaluate.py                     # float32 vs int8

Licencia: los modelos de InsightFace son solo para investigacion no comercial,
y los .onnx/.tflite derivados heredan esa restriccion (por eso models/ y work/
estan en .gitignore).
