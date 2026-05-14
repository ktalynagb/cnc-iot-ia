/**
 * cnc_iot_esp32.ino — FLUX CNC IoT · Recolección de datos etiquetados
 * =======================================================================
 * Captura datos de vibración (MPU-6050) + temperatura/humedad (DHT11)
 * y los envía por Serial en formato CSV para ser guardados en el PC
 * con el script captura_datos.py
 *
 * Hardware:
 *   ESP32-C3 Super Mini
 *   MPU-6050  SDA=GPIO8  SCL=GPIO9  (I2C, dirección 0x68)
 *   DHT22     GPIO0
 *
 * Formato de salida Serial (CSV):
 *   timestamp_ms,accel_x,accel_y,accel_z,temperatura,humedad
 *
 * Frecuencia de muestreo: ~100 Hz (10 ms por muestra)
 * Baudrate: 115200
 *
 * Protocolo de control (desde Python):
 *   'S' → iniciar/reanudar envío de datos
 *   'X' → pausar envío de datos
 */

#include <Wire.h>
#include <DHT.h>

// ── Pines ──────────────────────────────────────────────────────────────────
#define DHT_PIN         0
#define DHT_TYPE        DHT22
#define MPU_ADDR        0x68
#define SDA_PIN         8
#define SCL_PIN         9

// ── Configuración ──────────────────────────────────────────────────────────
#define SAMPLE_DELAY_MS  10    // 10 ms → 100 Hz
#define DHT_INTERVAL_MS  2000  // DHT11 se actualiza cada 2 s

DHT dht(DHT_PIN, DHT_TYPE);

float temp = 25.0f;
float hum  = 50.0f;
unsigned long lastDHT = 0;
bool streaming        = false;

// ────────────────────────────────────────────────────────────────────────────
void mpuInit() {
  // Despertar MPU-6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Rango ±2g (máxima sensibilidad para detectar vibraciones)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission(true);
}

void mpuReadAccel(float* ax, float* ay, float* az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);

  int16_t rx = (Wire.read() << 8) | Wire.read();
  int16_t ry = (Wire.read() << 8) | Wire.read();
  int16_t rz = (Wire.read() << 8) | Wire.read();

  // Convertir LSB → m/s² (escala ±2g: 16384 LSB/g)
  const float scale = 9.81f / 16384.0f;
  *ax = rx * scale;
  *ay = ry * scale;
  *az = rz * scale;
}

// ────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Wire.begin(SDA_PIN, SCL_PIN);

  // Verificar que el MPU-6050 responde en el bus I2C
  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() == 0) {
    Serial.println("#MPU-6050 detectado OK (0x68)");
    mpuInit();
  } else {
    Serial.println("#ERROR: MPU-6050 no encontrado. Revisa SDA=GPIO8, SCL=GPIO9 y alimentacion 3.3V");
  }

  dht.begin();

  // Primera lectura DHT11 (necesita ~2 s después de encender)
  delay(2000);
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (!isnan(t) && !isnan(h)) {
    temp = t; hum = h;
    Serial.println("#DHT11 detectado OK");
  } else {
    Serial.println("#ERROR: DHT11 sin respuesta. Revisa conexion en GPIO0 y resistencia pull-up 10k");
  }
  lastDHT = millis();

  Serial.println("#FLUX CNC - Firmware de captura de datos");
  Serial.println("#Formato: timestamp_ms,accel_x,accel_y,accel_z,temperatura,humedad");
  Serial.println("#Enviar 'S' para iniciar | 'X' para pausar");
  Serial.println("#LISTO");
}

// ────────────────────────────────────────────────────────────────────────────
void loop() {
  // Procesar comandos del PC
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'S') {
      streaming = true;
    } else if (cmd == 'X') {
      streaming = false;
    }
  }

  if (!streaming) {
    delay(50);
    return;
  }

  unsigned long now = millis();

  // Actualizar DHT11 (lectura lenta, no bloquea el loop principal)
  if (now - lastDHT >= DHT_INTERVAL_MS) {
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t) && !isnan(h)) { temp = t; hum = h; }
    lastDHT = now;
  }

  float ax, ay, az;
  mpuReadAccel(&ax, &ay, &az);

  // Línea CSV: timestamp_ms,ax,ay,az,temp,hum
  Serial.print(now);      Serial.print(',');
  Serial.print(ax, 4);    Serial.print(',');
  Serial.print(ay, 4);    Serial.print(',');
  Serial.print(az, 4);    Serial.print(',');
  Serial.print(temp, 2);  Serial.print(',');
  Serial.println(hum, 2);

  delay(SAMPLE_DELAY_MS);
}
