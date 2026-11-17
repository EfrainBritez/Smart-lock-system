# C3a: embedding facial calculado dentro del Grove Vision AI V2

El firmware `tflm_fd_fm` ya detecta rostros. Este parche le añade un cuarto modelo,
la MobileFaceNet int8 compilada con Vela, que corre en la NPU y devuelve un vector de
512 valores int8 por rostro.

    plano Y (gris 320x240) -> recorte CUADRADO alrededor de la caja del detector (1.25x)
    -> 112x112 -> Y replicado a 3 canales -> NPU -> vector int8[512] -> JSON "embedding"

Sin alineación con landmarks: es la línea base. C3b mide cuánto cuesta eso.
La malla facial y el iris siguen corriendo como antes (así el JSON conserva su forma).

## Qué está verificado y qué no

- Verificado: el parche aplica sobre el commit `d3265e20...`, el firmware compila y enlaza,
  la imagen mide 393 KB (límite 1 MB) y el modelo en flash queda en 0x400000-0x718280.
  La herramienta de Python se probó con eventos sintéticos.
- NO verificado (no tengo el módulo): que el modelo corra en la NPU, que el recorte y la
  reducción del firmware coincidan con la réplica de Python, el tiempo real por rostro
  (Vela estima ~177 ms) y que la SRAM alcance (mi estimación deja ~360 KiB libres).

## Pasos

### 1. Recompilar el modelo con la configuración oficial de Vela (carpeta C1, Windows o WSL)

    python 04_vela.py

Usa el `vela_config.ini` nuevo (el oficial de Seeed). El resultado es
`work/vela/embedding_fp32_full_integer_quant_vela.tflite` (~3.24 MB).

### 2. Aplicar el parche y compilar (WSL)

    python3 apply_c3a.py --sdk ~/Seeed_Grove_Vision_AI_Module_V2
    ./build_fd_fm.sh

Si el parche dice que un ancla aparece 0 veces, tu SDK no es el commit probado: avísame.
Para deshacerlo: `python3 apply_c3a.py --sdk ~/Seeed_Grove_Vision_AI_Module_V2 --revert`.

### 3. Llevar a Windows el firmware y el modelo

El modelo debe llamarse con su dirección de flash, para que `flash_fd_fm.py` lo detecte:

    S=~/Seeed_Grove_Vision_AI_Module_V2
    D=/mnt/c/himax/Seeed_Grove_Vision_AI_Module_V2
    cp $S/we2_image_gen_local/output_case1_sec_wlcsp/output.img $D/we2_image_gen_local/output_case1_sec_wlcsp/
    cp "/mnt/c/Users/Efrain/Documents/Casptone/Wio terminal/C1_/work/vela/embedding_fp32_full_integer_quant_vela.tflite" \
       $D/model_zoo/tflm_fd_fm/3_embedding_0x400000.tflite

(Ajusta la ruta del modelo a donde esté tu carpeta C1.)

### 4. Grabar (Windows)

    python flash_fd_fm.py COM7 --sdk C:\himax\Seeed_Grove_Vision_AI_Module_V2 --dry-run

Debe listar **4** modelos (los 3 de siempre + `3_embedding_0x400000.tflite`). Después,
sin `--dry-run`. La transferencia es más larga que antes porque el modelo pesa 3.2 MB.
Pulsa RESET al terminar.

### 5. Comprobar el arranque (PuTTY, 921600 baudios, ANTES de pulsar RESET)

Busca una línea que empiece por `EMB:`:

| Línea | Significado |
|---|---|
| `EMB: modelo listo (in scale=7843 e-6, zp=-1 \| out scale=... e-6, zp=...)` | Todo bien. |
| `EMB: no hay modelo TFLite en 0x400000; embeddings desactivados` | El modelo no se grabó: revisa el paso 4. El resto del firmware funciona. |
| `EMB: AllocateTensors fallo (arena insuficiente?)` | Mándame el mensaje completo. |
| `EMB: tipo o tamano de tensores inesperado (...)` | El modelo no es el esperado (112x112x3 -> 512 int8). |
| No imprime nada o se reinicia en bucle | Mándame lo que veas; se restaura con `--restore`. |

### 6. Validar los vectores (cierra PuTTY antes)

    python device_embeddings.py watch COM7 --onnx C:\ruta\C1_\models\embedding_fp32.onnx

Pon un rostro a 40-60 cm. Cada línea compara el vector del módulo con el que calcula
Python (float32) sobre el MISMO recorte. Esperable: cos alto (~0.9 o más). Por debajo de
~0.8 indica un problema (orden de canales, cuantización, recorte...). Al final muestra el
tick medio del embedding en ciclos de CPU.

### 7. C3b: precisión real (misma persona vs distinta)

    python device_embeddings.py collect COM7 Ana --count 30
    python device_embeddings.py collect COM7 Luis --count 30
    python device_embeddings.py evaluate

Varía el ángulo y la distancia mientras captura. Compara con `05_evaluate.py` de la etapa A.