"""
model/export_weights.py — FLUX CNC IoT · Edge AI
==================================================
Genera automáticamente el archivo model_weights.h
con los pesos del modelo entrenado en Python.

Uso (después de entrenar el modelo):
    python model/export_weights.py

Requiere:
    - modelo entrenado guardado como 'model/mlp_model.pkl'
      (o adaptar la carga según el formato que use David)

Salida:
    - firmware/inference/model_weights.h  (listo para copiar al ESP32)
"""

import numpy as np
import joblib
import os

# ── Cargar modelo entrenado ───────────────────────────────────────────────────
MODEL_PATH  = "model/mlp_model.pkl"        # Ajustar según formato de David
SCALER_PATH = "model/scaler.pkl"           # StandardScaler de sklearn
OUTPUT_PATH = "firmware/inference/model_weights.h"

print(f"Cargando modelo desde {MODEL_PATH}...")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Extraer pesos ─────────────────────────────────────────────────────────────
# Asume sklearn MLPClassifier con 1 capa oculta
W1     = model.coefs_[0].T      # shape: (16, 8)
b1     = model.intercepts_[0]   # shape: (16,)
W2     = model.coefs_[1].T      # shape: (3, 16)
b2     = model.intercepts_[1]   # shape: (3,)

# Parámetros de normalización
mean   = scaler.mean_            # shape: (8,)
std    = scaler.scale_           # shape: (8,)

print(f"W1: {W1.shape}  b1: {b1.shape}")
print(f"W2: {W2.shape}  b2: {b2.shape}")
print(f"Normalización: mean={mean.round(4)}  std={std.round(4)}")

# ── Función auxiliar para formatear arrays C++ ────────────────────────────────
def fmt_1d(arr, name, dtype="float"):
    vals = ", ".join(f"{v:.6f}f" for v in arr)
    return f"const {dtype} {name}[{len(arr)}] = {{\n  {vals}\n}};\n"

def fmt_2d(arr, name, dtype="float"):
    rows = []
    for row in arr:
        rows.append("  { " + ", ".join(f"{v:.6f}f" for v in row) + " }")
    inner = ",\n".join(rows)
    r, c = arr.shape
    return f"const {dtype} {name}[{r}][{c}] = {{\n{inner}\n}};\n"

# ── Generar archivo .h ────────────────────────────────────────────────────────
header = f"""/**
 * model_weights.h — FLUX CNC IoT · Edge AI
 * ==========================================
 * ARCHIVO GENERADO AUTOMÁTICAMENTE por export_weights.py
 * NO EDITAR MANUALMENTE
 *
 * Arquitectura: {W1.shape[1]} entradas → {W1.shape[0]} neuronas ocultas → {W2.shape[0]} salidas
 *
 * Features (orden de las {W1.shape[1]} entradas):
 *   [0] media(accel_x)   [1] varianza(accel_x)
 *   [2] media(accel_y)   [3] varianza(accel_y)
 *   [4] media(accel_z)   [5] varianza(accel_z)
 *   [6] temperatura      [7] humedad
 *
 * Clases de salida:
 *   [0] Reposo   [1] Operacion Normal   [2] Anomalia
 */

#pragma once

"""

content  = header
content += "// ── Parámetros de normalización Z-score ──\n"
content += fmt_1d(mean, "NORM_MEAN")
content += "\n"
content += fmt_1d(std,  "NORM_STD")
content += "\n"
content += "// ── Pesos capa 1: W1[16][8] ──\n"
content += fmt_2d(W1, "W1")
content += "\n"
content += "// ── Bias capa 1: b1[16] ──\n"
content += fmt_1d(b1, "BIAS_1")
content += "\n"
content += "// ── Pesos capa 2: W2[3][16] ──\n"
content += fmt_2d(W2, "W2")
content += "\n"
content += "// ── Bias capa 2: b2[3] ──\n"
content += fmt_1d(b2, "BIAS_2")
content += "\n"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    f.write(content)

print(f"\n✓ Archivo generado: {OUTPUT_PATH}")
print("  Copia el archivo al Arduino IDE y compila el firmware.")
