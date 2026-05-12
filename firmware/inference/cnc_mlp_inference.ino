/**
 * cnc_mlp_inference.ino — FLUX CNC IoT · Edge AI
 * ================================================
 * Clasificador de estados vibracionales MLP en tiempo real.
 * Arquitectura: Entrada(8) → Oculta(16) → Salida(3)
 *
 * Features (8 entradas):
 *   [0] media(accel_x)    [1] varianza(accel_x)
 *   [2] media(accel_y)    [3] varianza(accel_y)
 *   [4] media(accel_z)    [5] varianza(accel_z)
 *   [6] temperatura       [7] humedad
 *
 * Clases:
 *   0 = Reposo
 *   1 = Operación Normal
 *   2 = Anomalía / Vibración Excesiva
 *
 * Hardware:
 *   ESP32-C3 Super Mini
 *   MPU-6050  → SDA=GPIO8, SCL=GPIO9
 *   DHT22     → GPIO0
 *
 * Dependencias:
 *   - Adafruit DHT sensor library
 *   - Wire.h (incluida en ESP32 core)
 *
 * Los pesos (model_weights.h) son generados por David
 * tras el entrenamiento en Python/Google Colab.
 */

#include <Wire.h>
#include <DHT.h>
#include "model_weights.h"
#include "mlp_inference.h"

// ── Pines ──────────────────────────────────────────────────────────────────
#define DHT_PIN       0
#define DHT_TYPE      DHT22
#define MPU_ADDR      0x68
#define LED_PIN       10    // Indicador de alerta

// ── Configuración de ventana ───────────────────────────────────────────────
#define WINDOW_SIZE   32    // Muestras para calcular media y varianza
#define SAMPLE_DELAY  10    // ms entre muestras MPU (~100 Hz)

// ── Objetos ────────────────────────────────────────────────────────────────
DHT dht(DHT_PIN, DHT_TYPE);

// ── Buffers de ventana deslizante ──────────────────────────────────────────
float buf_x[WINDOW_SIZE];
float buf_y[WINDOW_SIZE];
float buf_z[WINDOW_SIZE];
int   buf_idx   = 0;
bool  buf_ready = false;

// ── Etiquetas de clase ─────────────────────────────────────────────────────
const char* LABELS[] = { "REPOSO", "OPERACION_NORMAL", "ANOMALIA" };

// ────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("=== FLUX CNC — MLP Edge Inference ===");

  // Inicializar I2C y MPU-6050
  Wire.begin(8, 9);   // SDA=8, SCL=9 para ESP32-C3
  mpu_init();

  // Inicializar DHT22
  dht.begin();

  // LED de alerta
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println("Hardware listo. Recolectando ventana inicial...");
}

// ────────────────────────────────────────────────────────────────────────────
void loop() {
  // 1. Leer muestra del MPU-6050
  float ax, ay, az;
  mpu_read_accel(&ax, &ay, &az);

  // 2. Guardar en buffer circular
  buf_x[buf_idx] = ax;
  buf_y[buf_idx] = ay;
  buf_z[buf_idx] = az;
  buf_idx = (buf_idx + 1) % WINDOW_SIZE;
  if (buf_idx == 0) buf_ready = true;

  // 3. Cuando la ventana esté llena → inferencia
  if (buf_ready) {
    // Calcular features estadísticos de la ventana
    float features[NUM_INPUTS];
    compute_features(buf_x, buf_y, buf_z, WINDOW_SIZE, features);

    // Leer temperatura y humedad (sensores lentos — no en cada muestra)
    static float temp = 25.0f, hum = 50.0f;
    static unsigned long last_dht = 0;
    if (millis() - last_dht > 2000) {
      float t = dht.readTemperature();
      float h = dht.readHumidity();
      if (!isnan(t) && !isnan(h)) { temp = t; hum = h; }
      last_dht = millis();
    }
    features[6] = temp;
    features[7] = hum;

    // Normalizar features (parámetros calculados en entrenamiento)
    normalize_features(features);

    // Inferencia MLP
    float probs[NUM_OUTPUTS];
    mlp_forward(features, probs);

    // Clase predicha
    int clase = argmax(probs, NUM_OUTPUTS);

    // 4. Mostrar resultado por Serial
    Serial.println("──────────────────────────────");
    Serial.printf("Temp: %.2f°C  Hum: %.2f%%\n", temp, hum);
    Serial.printf("Accel media  X:%.3f Y:%.3f Z:%.3f\n",
                  features[0], features[2], features[4]);
    Serial.printf("Accel var    X:%.4f Y:%.4f Z:%.4f\n",
                  features[1], features[3], features[5]);
    Serial.printf("Prediccion: [%d] %s\n", clase, LABELS[clase]);
    Serial.printf("Probs: R=%.2f ON=%.2f AN=%.2f\n",
                  probs[0], probs[1], probs[2]);

    // 5. Activar LED si hay anomalía
    digitalWrite(LED_PIN, clase == 2 ? HIGH : LOW);

    // Reiniciar ventana
    buf_ready = false;
  }

  delay(SAMPLE_DELAY);
}

// ────────────────────────────────────────────────────────────────────────────
// Inicializar MPU-6050
void mpu_init() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);   // PWR_MGMT_1
  Wire.write(0x00);   // Despertar el MPU
  Wire.endTransmission(true);

  // Rango ±2g (más sensible para detectar vibraciones finas)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);   // ACCEL_CONFIG
  Wire.write(0x00);   // ±2g
  Wire.endTransmission(true);

  Serial.println("MPU-6050 inicializado (±2g)");
}

// ────────────────────────────────────────────────────────────────────────────
// Leer aceleración del MPU-6050 en m/s²
void mpu_read_accel(float* ax, float* ay, float* az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);   // Registro ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);

  int16_t raw_x = (Wire.read() << 8) | Wire.read();
  int16_t raw_y = (Wire.read() << 8) | Wire.read();
  int16_t raw_z = (Wire.read() << 8) | Wire.read();

  // Escala para ±2g: 16384 LSB/g → convertir a m/s²
  const float scale = 9.81f / 16384.0f;
  *ax = raw_x * scale;
  *ay = raw_y * scale;
  *az = raw_z * scale;
}

// ────────────────────────────────────────────────────────────────────────────
// Calcular media y varianza de cada eje
void compute_features(float* bx, float* by, float* bz,
                      int n, float* features) {
  float sum_x=0, sum_y=0, sum_z=0;
  for (int i=0; i<n; i++) {
    sum_x += bx[i]; sum_y += by[i]; sum_z += bz[i];
  }
  float mean_x = sum_x/n, mean_y = sum_y/n, mean_z = sum_z/n;

  float var_x=0, var_y=0, var_z=0;
  for (int i=0; i<n; i++) {
    var_x += (bx[i]-mean_x)*(bx[i]-mean_x);
    var_y += (by[i]-mean_y)*(by[i]-mean_y);
    var_z += (bz[i]-mean_z)*(bz[i]-mean_z);
  }
  features[0] = mean_x;  features[1] = var_x/n;
  features[2] = mean_y;  features[3] = var_y/n;
  features[4] = mean_z;  features[5] = var_z/n;
}

// ────────────────────────────────────────────────────────────────────────────
// Índice del valor máximo (argmax)
int argmax(float* arr, int n) {
  int idx = 0;
  for (int i=1; i<n; i++) if (arr[i] > arr[idx]) idx = i;
  return idx;
}
