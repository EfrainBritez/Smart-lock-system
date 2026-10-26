#!/usr/bin/env bash
# C2 - Compila el ejemplo tflm_fd_fm del SDK de Himax y genera output.img
# Linux / WSL2 / macOS. Requiere: git, make, python3 y el toolchain
# "Arm GNU Toolchain 13.2.Rel1 (arm-none-eabi)" en el PATH.
#
# Uso:  ./build_fd_fm.sh                 (compilacion en serie, mas segura)
#       JOBS=4 ./build_fd_fm.sh          (en paralelo; sin probar)
#       APP_TYPE=tflm_yolov8_gender_cls ./build_fd_fm.sh
set -euo pipefail

SDK_DIR="${SDK_DIR:-$HOME/Seeed_Grove_Vision_AI_Module_V2}"
APP_TYPE="${APP_TYPE:-tflm_fd_fm}"
JOBS="${JOBS:-1}"
ELF_REL="obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf"

if ! command -v arm-none-eabi-gcc >/dev/null; then
  cat <<'MSG'
No encuentro arm-none-eabi-gcc. Instala el Arm GNU Toolchain 13.2.Rel1:

  cd $HOME
  wget https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz
  tar -xvf arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz
  export PATH="$HOME/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-eabi/bin/:$PATH"
MSG
  exit 1
fi
echo "Toolchain: $(arm-none-eabi-gcc --version | head -1)"

if [ ! -d "$SDK_DIR" ]; then
  git clone --recursive https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2.git "$SDK_DIR"
fi

cd "$SDK_DIR/EPII_CM55M_APP_S"
# Solo reemplaza el nombre (respeta los saltos de linea CRLF del archivo)
sed -i -E "s/^APP_TYPE *= *[A-Za-z0-9_]+/APP_TYPE = ${APP_TYPE}/" makefile
grep -n "^APP_TYPE" makefile | tr -d '\r'

make -j"$JOBS"

echo
echo "=== Tamano del firmware ==="
arm-none-eabi-size "$ELF_REL"

cd "$SDK_DIR/we2_image_gen_local"
cp "../EPII_CM55M_APP_S/$ELF_REL" input_case1_secboot/
if [ "$(uname -s)" = "Darwin" ]; then GEN=./we2_local_image_gen_macOS_arm64; else GEN=./we2_local_image_gen; fi
chmod +x "$GEN"
"$GEN" project_case1_blp_wlcsp.json

echo
echo "Imagen lista: $SDK_DIR/we2_image_gen_local/output_case1_sec_wlcsp/output.img (maximo 1 MB)"
ls -la output_case1_sec_wlcsp/output.img
