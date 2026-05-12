/**
 * scaler_params.h — FLUX CNC IoT · Edge AI
 * ==========================================
 * Parámetros de normalización Z-score.
 * GENERADO AUTOMÁTICAMENTE por model/export_weights.py
 * NO EDITAR MANUALMENTE
 *
 * Features (orden):
 *   [0] media(accel_x)   [1] varianza(accel_x)
 *   [2] media(accel_y)   [3] varianza(accel_y)
 *   [4] media(accel_z)   [5] varianza(accel_z)
 *   [6] temperatura      [7] humedad
 */

#pragma once

// REEMPLAZAR con valores reales del export_weights.py
const float SCALER_MEAN[8] = {
  0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 25.0f, 60.0f
};

const float SCALER_STD[8] = {
  1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 5.0f, 10.0f
};
