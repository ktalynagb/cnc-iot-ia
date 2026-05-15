/**
 * cnc_mlp_inference.ino — FLUX CNC IoT · Edge AI (TF Lite Micro)
 * ================================================================
 * Clasificador de estados vibracionales MLP en tiempo real.
 * Usa TensorFlow Lite Micro para la inferencia en el ESP32-C3.
 *
 * Arquitectura: Entrada(8) → Oculta(16) → Salida(3)
 *
 * Features (8 entradas):
 * [0] media(accel_x)    [1] varianza(accel_x)
 * [2] media(accel_y)    [3] varianza(accel_y)
 * [4] media(accel_z)    [5] varianza(accel_z)
 * [6] temperatura       [7] humedad
 *
 * Clases:
 * 0 = Reposo  |  1 = Operación Normal  |  2 = Anomalía
 */

#include <Wire.h>
#include <DHT.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "model.h"
#include "scaler_params.h"

// ── Pines ──────────────────────────────────────────────────────────────────
#define DHT_PIN       0
#define DHT_TYPE      DHT22
#define MPU_ADDR      0x68
#define LED_PIN       10   

// ── Configuración ──────────────────────────────────────────────────────────
#define WINDOW_SIZE   32
#define SAMPLE_DELAY  10
#define NUM_INPUTS     8
#define NUM_OUTPUTS    3

// ── TF Lite Micro (Globales) ───────────────────────────────────────────────
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tf_model       = nullptr;
  tflite::MicroInterpreter* interpreter    = nullptr;
  TfLiteTensor* input_tensor   = nullptr;
  TfLiteTensor* output_tensor  = nullptr;

  static tflite::MicroErrorReporter micro_error_reporter;
  static tflite::AllOpsResolver     resolver;

  constexpr int TENSOR_ARENA_SIZE = 8 * 1024;
  alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
}

// ── Objetos y buffers ──────────────────────────────────────────────────────
DHT dht(DHT_PIN, DHT_TYPE);
float buf_x[WINDOW_SIZE], buf_y[WINDOW_SIZE], buf_z[WINDOW_SIZE];
int   buf_idx   = 0;
bool  buf_ready = false;
const char* LABELS[] = { "REPOSO", "OPERACION_NORMAL", "ANOMALIA" };

// ── Prototipos ─────────────────────────────────────────────────────────────
void mpu_init();
void mpu_read_accel(float*, float*, float*);
void compute_features(float*, float*, float*, int, float*);
void normalize_features(float*);
int  argmax(float*, int);

// ────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n=== FLUX CNC IoT - Edge AI Inference ===");

  error_reporter = &micro_error_reporter;

  tf_model = tflite::GetModel(g_model);
  if (tf_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, 
      "Modelo con versión %d no compatible con schema %d",
      tf_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  Serial.println("[TFLite] Modelo cargado correctamente.");

  static tflite::MicroInterpreter static_interpreter(
      tf_model, 
      resolver, 
      tensor_arena, 
      TENSOR_ARENA_SIZE, 
      error_reporter
  );
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors() falló");
    return;
  }
  Serial.println("[TFLite] Tensores asignados.");

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Wire.begin(8, 9); // SDA=GPIO8, SCL=GPIO9 para ESP32-C3
  mpu_init();
  dht.begin();
  pinMode(LED_PIN, OUTPUT);
  
  Serial.println("=== Sistema Listo para Inferencia ===");
}

// ────────────────────────────────────────────────────────────────────────────
void loop() {
  float ax, ay, az;
  mpu_read_accel(&ax, &ay, &az);
  
  buf_x[buf_idx] = ax;
  buf_y[buf_idx] = ay;
  buf_z[buf_idx] = az;
  buf_idx = (buf_idx + 1) % WINDOW_SIZE;
  
  if (buf_idx == 0) buf_ready = true;

  if (buf_ready) {
    float features[NUM_INPUTS];
    compute_features(buf_x, buf_y, buf_z, WINDOW_SIZE, features);
    
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

    normalize_features(features);
    
    for (int i = 0; i < NUM_INPUTS; i++) {
      input_tensor->data.f[i] = features[i];
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("ERROR: Invoke fallo"); 
      return;
    }

    float probs[NUM_OUTPUTS];
    for (int i = 0; i < NUM_OUTPUTS; i++) {
      probs[i] = output_tensor->data.f[i];
    }
    
    int clase = argmax(probs, NUM_OUTPUTS);

    Serial.println("──────────────────────────────");
    Serial.printf("Temp: %.2f C  Hum: %.2f%%\n", temp, hum);
    Serial.printf("Media   X:%.3f Y:%.3f Z:%.3f\n", features[0], features[2], features[4]);
    Serial.printf("Varianza X:%.4f Y:%.4f Z:%.4f\n", features[1], features[3], features[5]);
    Serial.printf("Prediccion: [%d] %s\n", clase, LABELS[clase]);
    Serial.printf("Probs: R=%.2f ON=%.2f AN=%.2f\n", probs[0], probs[1], probs[2]);
    
    digitalWrite(LED_PIN, clase == 2 ? HIGH : LOW);
    buf_ready = false;
  }

  delay(SAMPLE_DELAY);
}

// ────────────────────────────────────────────────────────────────────────────
void mpu_init() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0x00);
  Wire.endTransmission(true);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C); Wire.write(0x00);
  Wire.endTransmission(true);
  Serial.println("MPU-6050 OK (±2g)");
}

void mpu_read_accel(float* ax, float* ay, float* az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);
  int16_t rx=(Wire.read()<<8)|Wire.read();
  int16_t ry=(Wire.read()<<8)|Wire.read();
  int16_t rz=(Wire.read()<<8)|Wire.read();
  const float s = 9.81f/16384.0f;
  *ax=rx*s; *ay=ry*s; *az=rz*s;
}

void compute_features(float* bx, float* by, float* bz, int n, float* f) {
  float sx=0,sy=0,sz=0;
  for(int i=0;i<n;i++){sx+=bx[i];sy+=by[i];sz+=bz[i];}
  float mx=sx/n,my=sy/n,mz=sz/n;
  
  float vx=0,vy=0,vz=0;
  for(int i=0;i<n;i++){
    vx+=(bx[i]-mx)*(bx[i]-mx);
    vy+=(by[i]-my)*(by[i]-my);
    vz+=(bz[i]-mz)*(bz[i]-mz);
  }
  f[0]=mx;f[1]=vx/n;f[2]=my;f[3]=vy/n;f[4]=mz;f[5]=vz/n;
}

void normalize_features(float* f) {
  for(int i=0;i<NUM_INPUTS;i++)
    f[i]=(f[i]-SCALER_MEAN[i])/(SCALER_STD[i]+1e-8f);
}

int argmax(float* arr, int n) {
  int idx=0;
  for(int i=1;i<n;i++) if(arr[i]>arr[idx]) idx=i;
  return idx;
}