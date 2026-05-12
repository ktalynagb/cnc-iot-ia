/**
 * mlp_inference.h — FLUX CNC IoT · Edge AI
 * ==========================================
 * Forward pass de la red MLP en C++ puro.
 * Sin dependencias externas — corre en cualquier ESP32.
 *
 * Arquitectura: 8 → 16 → 3
 * Activación capa oculta: ReLU
 * Activación capa salida: Softmax
 */

#pragma once
#include <math.h>
#include "model_weights.h"

#define NUM_INPUTS   8
#define NUM_HIDDEN  16
#define NUM_OUTPUTS  3

// ── Activaciones ─────────────────────────────────────────────────────────────

inline float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

void softmax(float* x, int n) {
  float max_val = x[0];
  for (int i=1; i<n; i++) if (x[i] > max_val) max_val = x[i];

  float sum = 0.0f;
  for (int i=0; i<n; i++) {
    x[i] = expf(x[i] - max_val);   // Estabilidad numérica
    sum += x[i];
  }
  for (int i=0; i<n; i++) x[i] /= sum;
}

// ── Normalización (Z-score) ───────────────────────────────────────────────────
// Parámetros calculados en entrenamiento y exportados en model_weights.h

void normalize_features(float* features) {
  for (int i=0; i<NUM_INPUTS; i++) {
    features[i] = (features[i] - NORM_MEAN[i]) / (NORM_STD[i] + 1e-8f);
  }
}

// ── Forward pass ──────────────────────────────────────────────────────────────

void mlp_forward(float* input, float* output) {
  float hidden[NUM_HIDDEN];

  // Capa 1: entrada → oculta (W1: 16×8, b1: 16)
  for (int j=0; j<NUM_HIDDEN; j++) {
    float sum = BIAS_1[j];
    for (int i=0; i<NUM_INPUTS; i++) {
      sum += W1[j][i] * input[i];
    }
    hidden[j] = relu(sum);
  }

  // Capa 2: oculta → salida (W2: 3×16, b2: 3)
  for (int k=0; k<NUM_OUTPUTS; k++) {
    float sum = BIAS_2[k];
    for (int j=0; j<NUM_HIDDEN; j++) {
      sum += W2[k][j] * hidden[j];
    }
    output[k] = sum;
  }

  // Softmax en la salida
  softmax(output, NUM_OUTPUTS);
}
