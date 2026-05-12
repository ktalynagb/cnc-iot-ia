/**
 * model_weights.h — FLUX CNC IoT · Edge AI
 * ==========================================
 * Pesos de la MLP exportados desde Python tras el entrenamiento.
 *
 * INSTRUCCIONES PARA DAVID:
 * ─────────────────────────
 * Después de entrenar el modelo en Python/Google Colab,
 * ejecutar el script model/export_weights.py para generar
 * este archivo automáticamente con los pesos reales.
 *
 * Los valores actuales son PLACEHOLDERS — el modelo no
 * funcionará correctamente hasta que se reemplacen.
 *
 * Arquitectura: 8 entradas → 16 neuronas ocultas → 3 salidas
 *
 * Features (orden de las 8 entradas):
 *   [0] media(accel_x)   [1] varianza(accel_x)
 *   [2] media(accel_y)   [3] varianza(accel_y)
 *   [4] media(accel_z)   [5] varianza(accel_z)
 *   [6] temperatura      [7] humedad
 *
 * Clases de salida:
 *   [0] Reposo   [1] Operación Normal   [2] Anomalía
 */

#pragma once

// ── Parámetros de normalización Z-score ──────────────────────────────────────
// Media de cada feature calculada sobre el dataset de entrenamiento
const float NORM_MEAN[8] = {
  0.0f,   // media(accel_x) — REEMPLAZAR con valor real
  0.0f,   // varianza(accel_x)
  0.0f,   // media(accel_y)
  0.0f,   // varianza(accel_y)
  0.0f,   // media(accel_z)
  0.0f,   // varianza(accel_z)
  25.0f,  // temperatura (°C)
  60.0f   // humedad (%)
};

// Desviación estándar de cada feature
const float NORM_STD[8] = {
  1.0f,   // media(accel_x) — REEMPLAZAR con valor real
  1.0f,   // varianza(accel_x)
  1.0f,   // media(accel_y)
  1.0f,   // varianza(accel_y)
  1.0f,   // media(accel_z)
  1.0f,   // varianza(accel_z)
  5.0f,   // temperatura
  10.0f   // humedad
};

// ── Pesos capa 1: W1[16][8] ───────────────────────────────────────────────────
// Dimensiones: NUM_HIDDEN × NUM_INPUTS = 16 × 8
const float W1[16][8] = {
  // Fila j = pesos de la neurona j hacia las 8 entradas
  // REEMPLAZAR con valores del export_weights.py
  { 0.1f,  0.2f, -0.1f,  0.3f,  0.0f, -0.2f,  0.1f,  0.0f },
  {-0.2f,  0.1f,  0.3f, -0.1f,  0.2f,  0.1f, -0.1f,  0.2f },
  { 0.3f, -0.1f,  0.2f,  0.1f, -0.3f,  0.2f,  0.0f, -0.1f },
  {-0.1f,  0.3f, -0.2f,  0.2f,  0.1f, -0.1f,  0.2f,  0.1f },
  { 0.2f,  0.0f,  0.1f, -0.3f,  0.2f,  0.1f, -0.2f,  0.3f },
  {-0.3f,  0.2f,  0.0f,  0.1f, -0.1f,  0.3f,  0.1f, -0.2f },
  { 0.1f, -0.2f,  0.3f,  0.0f,  0.2f, -0.1f,  0.3f,  0.1f },
  {-0.2f,  0.1f, -0.1f,  0.2f,  0.0f,  0.2f, -0.3f,  0.0f },
  { 0.0f,  0.3f,  0.2f, -0.2f,  0.1f,  0.0f,  0.2f, -0.1f },
  { 0.2f, -0.1f,  0.0f,  0.3f, -0.2f,  0.1f, -0.1f,  0.2f },
  {-0.1f,  0.2f, -0.3f,  0.1f,  0.3f, -0.2f,  0.0f,  0.1f },
  { 0.3f,  0.0f,  0.1f, -0.1f,  0.1f,  0.3f, -0.2f,  0.0f },
  {-0.2f,  0.1f,  0.2f,  0.0f, -0.1f,  0.1f,  0.3f, -0.3f },
  { 0.1f, -0.3f,  0.0f,  0.2f,  0.2f, -0.1f,  0.1f,  0.2f },
  {-0.1f,  0.2f,  0.1f, -0.2f,  0.0f,  0.3f, -0.1f,  0.1f },
  { 0.2f,  0.1f, -0.2f,  0.1f,  0.3f,  0.0f,  0.2f, -0.2f }
};

// ── Bias capa 1: b1[16] ───────────────────────────────────────────────────────
const float BIAS_1[16] = {
  // REEMPLAZAR con valores del export_weights.py
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

// ── Pesos capa 2: W2[3][16] ───────────────────────────────────────────────────
// Dimensiones: NUM_OUTPUTS × NUM_HIDDEN = 3 × 16
const float W2[3][16] = {
  // REEMPLAZAR con valores del export_weights.py
  { 0.2f, -0.1f,  0.3f,  0.1f, -0.2f,  0.1f,  0.0f,  0.2f,
   -0.1f,  0.3f,  0.1f, -0.2f,  0.2f,  0.0f, -0.1f,  0.3f },
  {-0.1f,  0.2f, -0.2f,  0.3f,  0.1f, -0.1f,  0.2f, -0.3f,
    0.2f, -0.1f,  0.3f,  0.1f, -0.1f,  0.2f,  0.3f, -0.2f },
  { 0.3f,  0.1f, -0.1f, -0.2f,  0.2f,  0.3f, -0.2f,  0.1f,
   -0.2f,  0.0f, -0.1f,  0.3f,  0.2f, -0.3f,  0.1f,  0.0f }
};

// ── Bias capa 2: b2[3] ────────────────────────────────────────────────────────
const float BIAS_2[3] = {
  // REEMPLAZAR con valores del export_weights.py
  0.0f, 0.0f, 0.0f
};
