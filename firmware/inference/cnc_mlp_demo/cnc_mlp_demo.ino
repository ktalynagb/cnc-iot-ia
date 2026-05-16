/**
 * cnc_mlp_demo.ino — FLUX CNC IoT · Edge AI · Firmware de Demostración
 * =====================================================================
 * Ejecuta la MLP en el ESP32-C3 usando el conjunto de test del repositorio
 * (model/test.csv) en lugar de sensores físicos (MPU-6050 / DHT11).
 *
 * Propósito: verificar que el modelo TF Lite Micro produce exactamente las
 * mismas predicciones que el pipeline Python, sin necesidad de hardware de
 * captura conectado.
 *
 * Arquitectura: Entrada(8) → Densa(16, ReLU) → Densa(3, Softmax)
 *
 * Features (8 entradas):
 *   [0] media(accel_x)   [1] varianza(accel_x)
 *   [2] media(accel_y)   [3] varianza(accel_y)
 *   [4] media(accel_z)   [5] varianza(accel_z)
 *   [6] temperatura      [7] humedad
 *
 * Clases de salida:
 *   0 = Reposo  |  1 = Operación Normal  |  2 = Anomalía
 *
 * ── Cómo generar test_data.h desde model/test.csv ──────────────────────────
 *   Desde la raíz del repositorio, ejecutar:
 *
 *     python3 - <<'EOF'
 *     import csv, datetime
 *     rows = list(csv.DictReader(open("model/test.csv")))
 *     N = len(rows)
 *     feat_cols = ["mean_x","var_x","mean_y","var_y","mean_z","var_z","temperature","humidity"]
 *     with open("firmware/inference/test_data.h","w") as out:
 *         out.write(f"#pragma once\nconst int TEST_N = {N};\n")
 *         out.write(f"const float TEST_DATA[{N}][8] = {{\n")
 *         for i,r in enumerate(rows):
 *             vals = ", ".join(f"{float(r[c]):.8f}f" for c in feat_cols)
 *             out.write(f"  {{ {vals} }}{','if i<N-1 else ''}\n")
 *         out.write("};\n")
 *         out.write(f"const int TEST_LABEL[{N}] = {{ {','.join(r['label'] for r in rows)} }};\n")
 *     print(f"test_data.h: {N} muestras")
 *     EOF
 *
 * ── Dependencias Arduino ────────────────────────────────────────────────────
 *   - TFLite_ESP32  by Eloquent Arduino  (Library Manager)
 *   - (NO requiere DHT ni Wire — este firmware es solo demostración)
 *
 * ── Limitaciones de memoria (ESP32-C3 Super Mini) ───────────────────────────
 *   - TENSOR_ARENA_SIZE = 8 KB es suficiente para este modelo (8→16→3).
 *   - TEST_DATA con 47 muestras × 8 floats × 4 bytes = 1.5 KB en flash.
 *   - Si se aumenta TEST_N a >500, considerar declarar TEST_DATA con
 *     PROGMEM y leer con pgm_read_float_near() para ahorrar RAM.
 *   - En ESP32-C3 la flash total es 4 MB; 47 muestras no presentan problema.
 */

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "model.h"
#include "scaler_params.h"
#include "test_data.h"

// ── Macro de configuración ──────────────────────────────────────────────────
// true  → los valores en TEST_DATA ya están escalados (StandardScaler aplicado
//          en Python); NO se llama a normalize_features().
// false → los valores en TEST_DATA son raw; SÍ se aplica normalize_features().
// Por defecto: true, porque model/test.csv contiene features ya escaladas.
#ifndef DEMO_USE_SCALED_DATA
  #define DEMO_USE_SCALED_DATA true
#endif

// ── Configuración del demo ──────────────────────────────────────────────────
#define DEMO_DELAY_MS  500   // Pausa entre muestras (ms). Reducir para acelerar.
#define LED_PIN        10    // LED indicador de Anomalía (GPIO10, igual que cnc_mlp_inference.ino)
#define NUM_INPUTS      8
#define NUM_OUTPUTS     3

// ── TF Lite Micro (Globales) ───────────────────────────────────────────────
namespace {
  tflite::ErrorReporter*   error_reporter = nullptr;
  const tflite::Model*     tf_model       = nullptr;
  tflite::MicroInterpreter* interpreter   = nullptr;
  TfLiteTensor*            input_tensor   = nullptr;
  TfLiteTensor*            output_tensor  = nullptr;

  static tflite::MicroErrorReporter micro_error_reporter;
  static tflite::AllOpsResolver     resolver;

  // 8 KB es suficiente para el modelo 8→16→3 con float32
  constexpr int TENSOR_ARENA_SIZE = 8 * 1024;
  alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
}

// ── Etiquetas de clase ─────────────────────────────────────────────────────
const char* LABELS[] = { "REPOSO", "OPERACION_NORMAL", "ANOMALIA" };

// ── Prototipos ─────────────────────────────────────────────────────────────
void     normalize_features(float* f);
int      argmax(float* arr, int n);
bool     init_tflite();

// ────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n=== FLUX CNC · Demo MLP sobre conjunto de test ===");
  Serial.printf("Muestras en TEST_DATA : %d\n", TEST_N);
  Serial.printf("DEMO_USE_SCALED_DATA  : %s\n",
                DEMO_USE_SCALED_DATA ? "true (sin normalize)" : "false (con normalize)");
  Serial.printf("Delay entre muestras  : %d ms\n\n", DEMO_DELAY_MS);

  // Configurar LED de anomalía
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Inicializar TF Lite Micro
  if (!init_tflite()) {
    Serial.println("[ERROR] Fallo al inicializar TF Lite Micro. Bloqueado.");
    while (true) { delay(1000); }
  }

  Serial.println("[OK] Sistema listo. Iniciando inferencia sobre TEST_DATA...\n");
}

// ────────────────────────────────────────────────────────────────────────────
void loop() {
  // Contadores de métricas
  static int  muestra_idx = 0;
  static int  aciertos    = 0;

  // Cuando se acaban las muestras, imprimir resumen y esperar
  if (muestra_idx >= TEST_N) {
    Serial.println("\n========================================");
    Serial.println("  FIN DEL CONJUNTO DE TEST");
    Serial.printf("  Aciertos: %d / %d  (%.1f%%)\n",
                  aciertos, TEST_N, 100.0f * aciertos / TEST_N);
    Serial.println("========================================");
    Serial.println("Reiniciando en 10 s...\n");
    muestra_idx = 0;
    aciertos    = 0;
    delay(10000);
    return;
  }

  // ── Cargar features de la muestra actual ───────────────────────────────
  float features[NUM_INPUTS];
  for (int j = 0; j < NUM_INPUTS; j++) {
    features[j] = TEST_DATA[muestra_idx][j];
  }
  int etiqueta_true = TEST_LABEL[muestra_idx];

  // ── Normalización (solo si los datos son raw) ───────────────────────────
  // Si DEMO_USE_SCALED_DATA == false, aplicar Z-score con SCALER_MEAN/SCALER_STD
  #if !DEMO_USE_SCALED_DATA
    normalize_features(features);
  #endif

  // ── Cargar tensor de entrada ────────────────────────────────────────────
  for (int i = 0; i < NUM_INPUTS; i++) {
    input_tensor->data.f[i] = features[i];
  }

  // ── Ejecutar inferencia ─────────────────────────────────────────────────
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.printf("[ERROR] Invoke() falló en muestra %d\n", muestra_idx);
    muestra_idx++;
    return;
  }

  // ── Obtener probabilidades y clase predicha ─────────────────────────────
  float probs[NUM_OUTPUTS];
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    probs[i] = output_tensor->data.f[i];
  }
  int etiqueta_pred = argmax(probs, NUM_OUTPUTS);

  // ── LED: encender si la predicción es ANOMALIA (clase 2) ───────────────
  digitalWrite(LED_PIN, etiqueta_pred == 2 ? HIGH : LOW);

  // ── Actualizar aciertos ─────────────────────────────────────────────────
  bool correcto = (etiqueta_pred == etiqueta_true);
  if (correcto) aciertos++;

  // ── Imprimir resultado por Serial ───────────────────────────────────────
  // Formato: idx | true | pred | correcto | P(R) P(ON) P(AN) | feat[0..7]
  Serial.printf("[%3d] true=%-16s pred=%-16s %s | P(R)=%.3f P(ON)=%.3f P(AN)=%.3f",
                muestra_idx,
                LABELS[etiqueta_true],
                LABELS[etiqueta_pred],
                correcto ? "OK  " : "FAIL",
                probs[0], probs[1], probs[2]);
  Serial.printf(" | feat: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                features[0], features[1], features[2], features[3],
                features[4], features[5], features[6], features[7]);

  muestra_idx++;
  delay(DEMO_DELAY_MS);
}

// ────────────────────────────────────────────────────────────────────────────
/**
 * init_tflite() — Inicializa TF Lite Micro exactamente como en
 *                 cnc_mlp_inference.ino. Retorna true si todo OK.
 */
bool init_tflite() {
  error_reporter = &micro_error_reporter;

  // Cargar modelo desde array C++ generado por export_weights.py
  tf_model = tflite::GetModel(g_model);
  if (tf_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
      "Versión de modelo %d incompatible con schema %d",
      tf_model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  Serial.println("[TFLite] Modelo cargado desde model.h");

  // Crear intérprete estático con arena de tensores
  static tflite::MicroInterpreter static_interpreter(
    tf_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter
  );
  interpreter = &static_interpreter;

  // Asignar tensores
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors() falló");
    return false;
  }

  // Obtener punteros a tensores de entrada y salida
  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.printf("[TFLite] Arena usada: %u bytes\n",
                (unsigned)interpreter->arena_used_bytes());
  return true;
}

// ────────────────────────────────────────────────────────────────────────────
/**
 * normalize_features() — Normalización Z-score con parámetros del scaler.
 * Solo se usa cuando DEMO_USE_SCALED_DATA == false (datos raw).
 * Idéntica a la función en cnc_mlp_inference.ino.
 */
void normalize_features(float* f) {
  for (int i = 0; i < NUM_INPUTS; i++) {
    f[i] = (f[i] - SCALER_MEAN[i]) / (SCALER_STD[i] + 1e-8f);
  }
}

// ────────────────────────────────────────────────────────────────────────────
/**
 * argmax() — Retorna el índice del elemento máximo en arr[0..n-1].
 * Idéntica a la función en cnc_mlp_inference.ino.
 */
int argmax(float* arr, int n) {
  int idx = 0;
  for (int i = 1; i < n; i++) {
    if (arr[i] > arr[idx]) idx = i;
  }
  return idx;
}
