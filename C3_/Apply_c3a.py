#!/usr/bin/env python3
"""
C3a - Anade al ejemplo tflm_fd_fm del SDK de Himax un cuarto modelo: el embedding
facial (MobileFaceNet int8 compilado con Vela) que corre en la NPU Ethos-U55.

Flujo en el firmware (por frame, solo si el detector encuentra un rostro):
  plano Y (gris 320x240) -> recorte CUADRADO alrededor de la caja del detector
  (1.25x el lado mayor) -> reduccion a 112x112 -> Y replicado a 3 canales,
  cuantizado con los parametros de entrada del modelo -> Invoke en la NPU ->
  vector int8 de 512 valores, enviado en el JSON "embedding" del evento INVOKE.

Si el modelo no esta grabado en flash, el firmware sigue funcionando como antes
(el JSON trae "enabled": 0).

Uso:
  python apply_c3a.py --sdk ~/Seeed_Grove_Vision_AI_Module_V2            # aplicar
  python apply_c3a.py --sdk ~/Seeed_Grove_Vision_AI_Module_V2 --revert   # deshacer
Probado contra el commit d3265e20d75fe20faccbf185d24454ad08b2fda8 del SDK.
"""
import argparse
import shutil
import sys
from pathlib import Path

MARK = "C3A_EMBEDDING"
APP = Path("EPII_CM55M_APP_S/app/scenario_app/tflm_fd_fm")
FILES = ["cvapp_fd_fm.cpp", "cvapp_fd_fm.h", "tflm_fd_fm.c",
         "common_config.h", "send_result.cpp", "send_result.h"]


def read(p):
    raw = p.read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n"), ("\r\n" if "\r\n" in raw else "\n")


def write(p, text, nl):
    p.write_bytes(text.replace("\n", nl).encode("utf-8"))


def once(text, old, new, what):
    n = text.count(old)
    if n != 1:
        sys.exit(f"[{what}] el texto ancla aparece {n} veces (se esperaba 1). "
                 "Tu SDK no coincide con el commit probado.")
    return text.replace(old, new)


# ---------------------------------------------------------------- bloques de codigo
EMB_BLOCK = r'''// ===== C3A_EMBEDDING: embedding facial en la NPU ============================
// Modelo: MobileFaceNet int8 compilado con Vela. Entrada 1x112x112x3, salida 1x512.
// El sensor entrega solo el plano Y (gris): se replica en los 3 canales.
#define EMB_INPUT_W     112
#define EMB_INPUT_H     112
#define EMB_DIM         512
// Lado del recorte cuadrado = EMB_CROP_NUM/EMB_CROP_DEN x el lado mayor de la caja.
#define EMB_CROP_NUM    5
#define EMB_CROP_DEN    4

static tflite::MicroInterpreter *emb_int_ptr = nullptr;
static TfLiteTensor *emb_input = nullptr;
static TfLiteTensor *emb_output = nullptr;
static int8_t   emb_lut[256];
static int8_t   emb_vec[EMB_DIM];
static float    emb_out_scale = 0.0f;
static int      emb_out_zp = 0;
static int      emb_valid = 0;
static int      emb_crop_x = 0, emb_crop_y = 0, emb_crop_side = 0;
static uint32_t emb_tick = 0;

static void emb_run(uint32_t raw_addr, uint32_t img_w, uint32_t img_h,
		int bx, int by, int bw, int bh)
{
	emb_valid = 0;
	if (emb_int_ptr == nullptr) {
		return;
	}

	uint32_t st_1, lc_1, st_2, lc_2;
	SystemGetTick(&st_1, &lc_1);

	// recorte cuadrado centrado en la caja, ajustado para quedar dentro de la imagen
	int big = (bw > bh) ? bw : bh;
	int side = (big * EMB_CROP_NUM) / EMB_CROP_DEN;
	int max_side = (int)((img_w < img_h) ? img_w : img_h);
	if (side > max_side) side = max_side;
	side &= ~1;
	if (side < 32) {
		return;
	}
	int x0 = bx + bw / 2 - side / 2;
	int y0 = by + bh / 2 - side / 2;
	if (x0 + side > (int)img_w) x0 = (int)img_w - side;
	if (y0 + side > (int)img_h) y0 = (int)img_h - side;
	if (x0 < 0) x0 = 0;
	if (y0 < 0) y0 = 0;
	x0 &= ~1;
	y0 &= ~1;

	hx_lib_image_copy_helium((uint8_t*)raw_addr, (uint8_t*)crop_img, img_w, img_h, 1,
			x0, y0, side, side);
	float sc = (float)(side - 1) / (float)(EMB_INPUT_W - 1);
	hx_lib_image_resize_helium((uint8_t*)crop_img, (uint8_t*)resized_img,
			side, side, 1, EMB_INPUT_W, EMB_INPUT_H, sc, sc);

	// Y -> 3 canales int8 (tabla de 256 entradas calculada al iniciar)
	int8_t *dst = emb_input->data.int8;
	const uint8_t *src = (const uint8_t*)resized_img;
	for (int i = 0; i < EMB_INPUT_W * EMB_INPUT_H; i++) {
		int8_t q = emb_lut[src[i]];
		dst[3 * i] = q;
		dst[3 * i + 1] = q;
		dst[3 * i + 2] = q;
	}

	if (emb_int_ptr->Invoke() != kTfLiteOk) {
		return;
	}
	memcpy(emb_vec, emb_output->data.int8, EMB_DIM);
	emb_crop_x = x0;
	emb_crop_y = y0;
	emb_crop_side = side;
	SystemGetTick(&st_2, &lc_2);
	emb_tick = (lc_2 - lc_1) * CPU_CLK + (st_1 - st_2);
	emb_valid = 1;
}
// ===== fin C3A_EMBEDDING =====================================================

'''

EMB_INIT = r'''	// ---- C3A_EMBEDDING: cargar el modelo de embeddings (opcional) ----
	{
		const uint8_t *emb_raw = (const uint8_t*)emb_model_addr;
		if (memcmp(emb_raw + 4, "TFL3", 4) != 0) {
			xprintf("EMB: no hay modelo TFLite en 0x%x; embeddings desactivados\n", emb_model_addr);
		} else {
			static const tflite::Model *EMB_model = tflite::GetModel((const void *)emb_model_addr);
			if (EMB_model->version() != TFLITE_SCHEMA_VERSION) {
				xprintf("EMB: version de esquema %d no soportada\n", (int)EMB_model->version());
			} else {
#if TFLM2209_U55TAG2205
				static tflite::MicroInterpreter emb_static_interpreter(EMB_model, op_resolver, (uint8_t*)tensor_arena, tensor_arena_size-(tensor_arena_model_tail_size*3), &micro_error_reporter);
#else
				static tflite::MicroInterpreter emb_static_interpreter(EMB_model, op_resolver, (uint8_t*)tensor_arena, tensor_arena_size-(tensor_arena_model_tail_size*3));
#endif
				if (emb_static_interpreter.AllocateTensors() != kTfLiteOk) {
					xprintf("EMB: AllocateTensors fallo (arena insuficiente?)\n");
				} else {
					TfLiteTensor *ti = emb_static_interpreter.input(0);
					TfLiteTensor *to = emb_static_interpreter.output(0);
					if (ti->type != kTfLiteInt8 || to->type != kTfLiteInt8
							|| ti->bytes != EMB_INPUT_W * EMB_INPUT_H * 3 || to->bytes != EMB_DIM) {
						xprintf("EMB: tipo o tamano de tensores inesperado (in %d B, out %d B)\n",
								(int)ti->bytes, (int)to->bytes);
					} else {
						float in_scale = ti->params.scale;
						int in_zp = ti->params.zero_point;
						for (int p = 0; p < 256; p++) {
							float v = ((float)p - 127.5f) / 127.5f;   // igual que ArcFace
							int q = (int)lrintf(v / in_scale) + in_zp;
							if (q > 127) q = 127;
							if (q < -128) q = -128;
							emb_lut[p] = (int8_t)q;
						}
						emb_out_scale = to->params.scale;
						emb_out_zp = to->params.zero_point;
						emb_input = ti;
						emb_output = to;
						emb_int_ptr = &emb_static_interpreter;
						xprintf("EMB: modelo listo (in scale=%d e-6, zp=%d | out scale=%d e-6, zp=%d)\n",
								(int)lrintf(in_scale * 1e6f), in_zp,
								(int)lrintf(emb_out_scale * 1e6f), emb_out_zp);
					}
				}
			}
		}
	}

'''

EMB_CALL = '''
		// C3A_EMBEDDING: embedding del rostro (recorte cuadrado desde el plano Y)
		emb_run(raw_addr, img_w, img_h,
				alg_fm_result->face_bbox[0].x, alg_fm_result->face_bbox[0].y,
				alg_fm_result->face_bbox[0].width, alg_fm_result->face_bbox[0].height);
'''

JSON_FN = r'''
// C3A_EMBEDDING: campo "embedding" del evento INVOKE. La escala (float) se envia como
// entero (e6) y tambien como sus 32 bits exactos, porque nano.specs no imprime floats.
std::string  emb_2_json_str(int enabled, int valid, const int8_t *vec, int n, float scale,
		int zp, int crop_x, int crop_y, int crop_side, uint32_t tick) {
	uint32_t scale_bits = 0;
	memcpy(&scale_bits, &scale, sizeof(scale_bits));
	std::string ss = "\"embedding\": {\"enabled\": ";
	ss += std::to_string(enabled);
	ss += ", \"valid\": ";
	ss += std::to_string(valid);
	ss += ", \"scale_e6\": ";
	ss += std::to_string((int)(scale * 1000000.0f + 0.5f));
	ss += ", \"scale_bits\": ";
	ss += std::to_string(scale_bits);
	ss += ", \"zp\": ";
	ss += std::to_string(zp);
	ss += ", \"crop\": [";
	ss += std::to_string(crop_x) + ", " + std::to_string(crop_y) + ", " + std::to_string(crop_side);
	ss += "], \"tick\": ";
	ss += std::to_string(tick);
	ss += ", \"data\": [";
	if (valid) {
		for (int i = 0; i < n; i++) {
			ss += std::to_string((int)vec[i]);
			if (i < n - 1) {
				ss += ", ";
			}
		}
	}
	ss += "]}";
	return ss;
}
'''


def apply(sdk):
    app = sdk / APP
    if not app.is_dir():
        sys.exit(f"No encuentro {app}")
    for f in FILES:
        if not (app / f).exists():
            sys.exit(f"Falta {app / f}")
    text = {f: read(app / f) for f in FILES}
    if any(MARK in t for t, _ in text.values()):
        sys.exit("El parche C3a ya esta aplicado (usa --revert para deshacerlo).")
    for f in FILES:  # copia de seguridad
        bak = app / (f + ".c3a.orig")
        if not bak.exists():
            shutil.copy2(app / f, bak)

    # common_config.h: direccion del modelo en flash (0x400000, despues del modelo il)
    t, nl = text["common_config.h"]
    import re
    m = re.search(r"^#define IRIS_LANDMARKS_FLASH_ADDR.*$", t, re.M)
    if not m:
        sys.exit("[common_config.h] no encuentro IRIS_LANDMARKS_FLASH_ADDR")
    t = t.replace(m.group(0), m.group(0) + "\n#define EMBEDDING_FLASH_ADDR  (BASE_ADDR_FLASH1_R_ALIAS+0x400000) // C3A_EMBEDDING", 1)
    text["common_config.h"] = (t, nl)

    # cvapp_fd_fm.h: nuevo parametro
    t, nl = text["cvapp_fd_fm.h"]
    t = once(t, "uint32_t il_model_addr)", "uint32_t il_model_addr, uint32_t emb_model_addr)", "cvapp_fd_fm.h")
    text["cvapp_fd_fm.h"] = (t + ("\n" if not t.endswith("\n") else "") + "// C3A_EMBEDDING\n", nl)

    # tflm_fd_fm.c: pasar la direccion
    t, nl = text["tflm_fd_fm.c"]
    t = once(t, "FACE_MESH_FLASH_ADDR, IRIS_LANDMARKS_FLASH_ADDR);",
             "FACE_MESH_FLASH_ADDR, IRIS_LANDMARKS_FLASH_ADDR, EMBEDDING_FLASH_ADDR); // C3A_EMBEDDING",
             "tflm_fd_fm.c")
    text["tflm_fd_fm.c"] = (t, nl)

    # send_result.h / .cpp: funcion de serializacion
    t, nl = text["send_result.h"]
    t = once(t, "std::string  algo_tick_2_json_str(uint32_t algo_tick);",
             "std::string  algo_tick_2_json_str(uint32_t algo_tick);\n"
             "std::string  emb_2_json_str(int enabled, int valid, const int8_t *vec, int n, float scale, "
             "int zp, int crop_x, int crop_y, int crop_side, uint32_t tick); // C3A_EMBEDDING",
             "send_result.h")
    text["send_result.h"] = (t, nl)

    t, nl = text["send_result.cpp"]
    start = t.index("std::string  algo_tick_2_json_str(uint32_t algo_tick)")
    end = t.index("\n}", start) + 2   # fin de la funcion (es la ultima del archivo)
    t = t[:end] + JSON_FN + t[end:]
    text["send_result.cpp"] = (t, nl)

    # cvapp_fd_fm.cpp
    t, nl = text["cvapp_fd_fm.cpp"]
    t = once(t, "constexpr int tensor_arena_size = 460*1024;//435*1024;",
             "constexpr int tensor_arena_size = 1210*1024; // C3A_EMBEDDING: Vela pide 1176 KiB (antes 460 KiB)",
             "arena")
    t = once(t, "int cv_fd_fm_init(bool security_enable", EMB_BLOCK + "int cv_fd_fm_init(bool security_enable", "bloque emb")
    t = once(t, "uint32_t il_model_addr) {", "uint32_t il_model_addr, uint32_t emb_model_addr) {", "firma init")
    t = once(t, '\txprintf("initial done\\n");', EMB_INIT + '\txprintf("initial done\\n");', "init emb")
    t = once(t, "\tint pad_sz = 0;\n", "\tint pad_sz = 0;\n\temb_valid = 0; // C3A_EMBEDDING\n", "reset emb")
    t = once(t, "\t\talg_fm_result->num_tracked_face_targets = alg_result->num_tracked_human_targets;\n",
             "\t\talg_fm_result->num_tracked_face_targets = alg_result->num_tracked_human_targets;\n" + EMB_CALL,
             "llamada emb")
    old = 'fm_point_results_2_json_str(el_fm_point_algo), ", ", img_2_json_str(&temp_el_jpg_img)));'
    new = ('fm_point_results_2_json_str(el_fm_point_algo), ", ", '
           'emb_2_json_str(emb_int_ptr != nullptr, emb_valid, emb_vec, EMB_DIM, emb_out_scale, emb_out_zp, '
           'emb_crop_x, emb_crop_y, emb_crop_side, emb_tick), ", ", img_2_json_str(&temp_el_jpg_img)));')
    t = once(t, old, new, "emision json")
    text["cvapp_fd_fm.cpp"] = (t, nl)

    for f, (t, nl) in text.items():
        write(app / f, t, nl)
    print("C3a aplicado a:", ", ".join(FILES))
    print("Copias originales: *.c3a.orig junto a cada archivo.")


def revert(sdk):
    app = sdk / APP
    n = 0
    for f in FILES:
        bak = app / (f + ".c3a.orig")
        if bak.exists():
            shutil.copy2(bak, app / f)
            bak.unlink()
            n += 1
    print(f"Restaurados {n} archivos.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aplica C3a al SDK de Himax")
    ap.add_argument("--sdk", required=True, help="carpeta Seeed_Grove_Vision_AI_Module_V2")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    sdk = Path(a.sdk).expanduser().resolve()
    revert(sdk) if a.revert else apply(sdk)