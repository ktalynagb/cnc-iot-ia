"""
model/export_weights.py — FLUX CNC IoT · Edge AI
==================================================
Exporta el modelo Keras entrenado a TF Lite y genera
los archivos necesarios para el firmware del ESP32.

Uso (después de entrenar):
    python model/export_weights.py

Salidas:
    model/model.tflite               → modelo TF Lite
    firmware/inference/model.h       → array C++ para Arduino
    firmware/inference/scaler_params.h → parámetros normalización
"""

import numpy as np
import tensorflow as tf
import joblib, os

MODEL_PATH    = "model/mlp_model.keras"
SCALER_PATH   = "model/scaler.pkl"
TFLITE_PATH   = "model/model.tflite"
MODEL_H_PATH  = "firmware/inference/model.h"
SCALER_H_PATH = "firmware/inference/scaler_params.h"

print("Cargando modelo Keras...")
model  = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Convertir a TF Lite ──────────────────────────────────────────────────
print("Convirtiendo a TF Lite (float32)...")
converter    = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

os.makedirs("model", exist_ok=True)
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"  -> {TFLITE_PATH} ({len(tflite_model)} bytes)")

# ── Generar model.h ──────────────────────────────────────────────────────
print("Generando model.h...")
os.makedirs(os.path.dirname(MODEL_H_PATH), exist_ok=True)

with open(TFLITE_PATH, "rb") as f:
    data = f.read()

hex_array = ", ".join(f"0x{b:02x}" for b in data)
model_h = f"""/**
 * model.h - FLUX CNC IoT - Edge AI
 * GENERADO AUTOMATICAMENTE por export_weights.py
 * NO EDITAR MANUALMENTE
 * Tamano del modelo: {len(data)} bytes
 */

#pragma once

const unsigned char g_model[] = {{
  {hex_array}
}};

const unsigned int g_model_len = {len(data)};
"""

with open(MODEL_H_PATH, "w") as f:
    f.write(model_h)
print(f"  -> {MODEL_H_PATH}")

# ── Generar scaler_params.h ──────────────────────────────────────────────
print("Generando scaler_params.h...")
mean = scaler.mean_
std  = scaler.scale_

def fmt_array(arr, name):
    vals = ", ".join(f"{v:.6f}f" for v in arr)
    return f"const float {name}[{len(arr)}] = {{\n  {vals}\n}};\n"

scaler_h = f"""/**
 * scaler_params.h - FLUX CNC IoT - Edge AI
 * GENERADO AUTOMATICAMENTE por export_weights.py
 * NO EDITAR MANUALMENTE
 */

#pragma once

{fmt_array(mean, "SCALER_MEAN")}
{fmt_array(std,  "SCALER_STD")}
"""

with open(SCALER_H_PATH, "w") as f:
    f.write(scaler_h)
print(f"  -> {SCALER_H_PATH}")

# ── Verificar ────────────────────────────────────────────────────────────
print("\nVerificando modelo TF Lite...")
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()
print(f"  Input:  {inp[0]['shape']}  {inp[0]['dtype']}")
print(f"  Output: {out[0]['shape']}  {out[0]['dtype']}")
print(f"\n  Exportacion completa! Tamano: {len(data)} bytes")
print("  Copia firmware/inference/ al Arduino IDE y compila.")
