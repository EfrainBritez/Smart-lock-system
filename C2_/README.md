# C2: compilar y cargar el ejemplo base (`tflm_fd_fm`) en el Grove Vision AI V2

Objetivo: confirmar que tu entorno puede **compilar y grabar** firmware propio
y ver el ejemplo de detección de rostros funcionando, SIN tocar todavía nuestro
modelo. Si esto funciona, C3 es "solo" cambiar un modelo.

> Esto reemplaza el firmware SSCMA/SenseCraft del módulo. Mientras tengas el
> firmware de Himax, `face_system.py --source grove` NO funcionará (usa los
> comandos AT de SSCMA). Se restaura en cualquier momento (ver abajo).

## Flujo recomendado en Windows (compilar en WSL2, grabar en Windows)

WSL2 no ve los puertos COM sin configuración extra (usbipd), así que:
**compila en WSL2 y graba desde Windows**.

### 1. Compilar (en WSL2 / Ubuntu)

    sudo apt update && sudo apt install -y git make python3
    cd $HOME
    wget https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz
    tar -xvf arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz
    export PATH="$HOME/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-eabi/bin/:$PATH"

    ./build_fd_fm.sh        # clona el SDK, compila (~10-30 min) y genera output.img

Usa el toolchain de Arm indicado y no el `gcc-arm-none-eabi` de `apt`: en mis
pruebas el de Ubuntu falló con un error de `PRIx64` (otra versión de newlib).

### 2. Llevar el SDK a Windows

Copia la carpeta `~/Seeed_Grove_Vision_AI_Module_V2` a Windows (por ejemplo
`C:\himax\Seeed_Grove_Vision_AI_Module_V2`), o clónala también en Windows y
copia solo `we2_image_gen_local/output_case1_sec_wlcsp/output.img`.

### 3. Grabar (en Windows, con Python)

    pip install -r C:\himax\Seeed_Grove_Vision_AI_Module_V2\xmodem\requirements.txt
    python flash_fd_fm.py COM5 --sdk C:\himax\Seeed_Grove_Vision_AI_Module_V2 --dry-run   # revisa el comando
    python flash_fd_fm.py COM5 --sdk C:\himax\Seeed_Grove_Vision_AI_Module_V2

- Cierra antes Tera Term, el Monitor Serie de Arduino y la pestaña de SenseCraft.
- Graba el firmware y los 3 modelos (detección, malla facial, iris).
- Al terminar pregunta si reinicias: pulsa el botón **RESET** del módulo.

### 4. Ver que funciona

Abre un terminal serial al puerto del módulo a **921600** baudios (Tera Term,
PuTTY o minicom). Deberías ver líneas de log por cada frame, y resultados
cuando hay un rostro delante de la cámara.

## Restaurar el firmware de fábrica

    python flash_fd_fm.py COM5 --sdk C:\himax\Seeed_Grove_Vision_AI_Module_V2 --restore

Graba la imagen `Seeed_SenseCraft_AI*.img` incluida en el SDK (es de dic. 2023).
Luego puedes usar SenseCraft Web Toolkit como antes.

## Presupuesto de memoria

    python memory_budget.py              # nuestro diseno sin pad_img
    python memory_budget.py --keep-pad   # conservando pad_img

Es una estimación hecha leyendo el código del SDK, no una medición en el módulo.

## Qué traerme de vuelta

1. La salida de `arm-none-eabi-size` que imprime `build_fd_fm.sh`.
2. Si compiló sin errores (si no, el primer error).
3. Si grabó bien y qué ves en el terminal serial tras el RESET.
