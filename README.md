# CNC IoT — Clasificador de Estados Vibracionales · Edge AI

Proyecto de la **Especialización en Inteligencia Artificial aplicada a IoT**  
Universidad Autónoma de Occidente · Práctica 4 — MLP en ESP32 (TinyML / Edge AI)

---

## Integrantes

| Rol | Persona | Responsabilidad |
|-----|---------|-----------------|
| Hardware & Recolección | Valentina | ESP32-C3 + MPU-6050 + DHT22 · Firmware de captura de datos etiquetados |
| Entrenamiento MLP | David | Dataset CSV · Entrenamiento Keras/TF · Export a TF Lite |
| Firmware & Despliegue | Ktalyna | Inferencia TF Lite Micro en ESP32 · Integración Edge AI |

---

## Caso de uso

**FLUX** es una empresa de maquinado CNC. El objetivo es detectar automáticamente el estado de la máquina en tiempo real, **directamente en el ESP32**, sin necesidad de enviar datos a la nube:

| Clase | Estado | Descripción |
|-------|--------|-------------|
| 0 | 🟢 Reposo | Máquina apagada o en pausa |
| 1 | 🔵 Operación Normal | Maquinado dentro de parámetros normales |
| 2 | 🔴 Anomalía | Vibración excesiva — posible falla mecánica |

---

## ¿Por qué Edge AI?

En las prácticas anteriores el ESP32 enviaba datos a la nube (Azure IoT Hub / Mosquitto) y el procesamiento ocurría en servidores remotos. En este proyecto la **inferencia ocurre directamente en el microcontrolador**:

```
Antes (Cloud AI):     ESP32 → Internet → Azure → Predicción → ESP32
Ahora (Edge AI):      ESP32 → Predicción local en < 1ms → Acción inmediata
```

**Ventajas para la CNC FLUX:**
- ⚡ Latencia < 1ms — reacción inmediata ante anomalías
- 🔌 Funciona sin internet — no depende de conectividad
- 🔒 Privacidad — los datos de vibración no salen de la máquina
- 💰 Costo cero en cómputo cloud

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────┐
│              ESP32-C3 Super Mini                 │
│                                                  │
│  MPU-6050 ──► Buffer circular (32 muestras)      │
│  DHT22    ──► Temperatura · Humedad              │
│                      │                           │
│              Feature Engineering                 │
│         media(x,y,z) · varianza(x,y,z)          │
│              temp · humedad = 8 features         │
│                      │                           │
│              Normalización Z-score               │
│                      │                           │
│         ┌────────────▼────────────┐              │
│         │  TF Lite Micro (MLP)   │  TinyML      │
│         │  8 → 16 → 3            │  Edge AI     │
│         │  ReLU      Softmax     │              │
│         └────────────┬────────────┘              │
│                      │                           │
│         REPOSO / OPERACION_NORMAL / ANOMALIA     │
│                      │                           │
│              Serial Monitor · LED                │
└─────────────────────────────────────────────────┘
```

---

## ML Pipeline (según clase Edge AI)

| Paso | Responsable | Descripción |
|------|------------|-------------|
| 1. Recolección de datos | Valentina + David | ESP32 graba CSV con etiquetas 0/1/2 |
| 2. Feature Engineering | David (Python) | Media y varianza de ventana de 32 muestras |
| 3. Entrenamiento MLP | David (Keras/Colab) | tf.keras.Sequential 8→16→3 |
| 4. Evaluación | David | Accuracy, Loss, matriz de confusión |
| 5. Exportación (Compilation) | David → `export_weights.py` | Keras → .tflite → model.h + scaler_params.h |
| 6. Despliegue Edge | Ktalyna | TF Lite Micro en ESP32 |

---

## Hardware

| Componente | Conexión | Función |
|------------|----------|---------|
| ESP32-C3 Super Mini | — | Microcontrolador. Corre la inferencia TF Lite Micro en tiempo real |
| MPU-6050 | SDA=GPIO8, SCL=GPIO9 | Acelerómetro I²C. Mide vibración X, Y, Z a ~100Hz |
| DHT22 | GPIO0 | Sensor temperatura (°C) y humedad (%) |
| LED (indicador) | GPIO10 | Se enciende cuando se detecta Anomalía (clase 2) |

⚠️ GPIO21 dañado — no usar.

---

## Features del modelo (8 entradas)

Sobre una ventana deslizante de **32 muestras** (~320ms a 100Hz):

| # | Feature | Cálculo |
|---|---------|---------|
| 0 | media(accel_x) | Promedio del eje X |
| 1 | varianza(accel_x) | Varianza del eje X |
| 2 | media(accel_y) | Promedio del eje Y |
| 3 | varianza(accel_y) | Varianza del eje Y |
| 4 | media(accel_z) | Promedio del eje Z |
| 5 | varianza(accel_z) | Varianza del eje Z |
| 6 | temperatura | Valor directo DHT22 (°C) |
| 7 | humedad | Valor directo DHT22 (%) |

> La temperatura y humedad se usan como valor directo porque el DHT22 mide lento (cada 2s) — calcular varianza no aporta información útil para detección de vibración.

---

## Estructura del repositorio

```
cnc-iot-ia/
├── Esp32/
│   └── cnc_iot_esp32/
│       ├── cnc_iot_esp32.ino     # Firmware recolección de datos (Valentina)
│       └── credentials.h         # WiFi (NO subir credenciales reales)
├── firmware/
│   └── inference/
│       ├── cnc_mlp_inference.ino # Firmware inferencia TF Lite Micro (Ktalyna)
│       ├── model.h               # Modelo TF Lite como array C++ (generado por David)
│       └── scaler_params.h       # Parámetros normalización Z-score (generado por David)
├── data/
│   ├── reposo.csv                # Dataset clase 0 (David)
│   ├── operacion_normal.csv      # Dataset clase 1 (David)
│   └── anomalia.csv              # Dataset clase 2 (David)
├── model/
│   ├── train_mlp.py              # Script entrenamiento Keras (David)
│   ├── export_weights.py         # Keras → .tflite → model.h + scaler_params.h (David)
│   ├── model.tflite              # Modelo exportado
│   ├── mlp_model.keras           # Modelo Keras guardado
│   └── scaler.pkl                # StandardScaler serializado
└── README.md
```

---

## Inicio rápido — Recolección de datos (Valentina)

El ESP32 graba los datos en un archivo CSV en el computador via Serial.
Capturar **200-350 muestras por clase** con el ESP32 en cada estado:

```bash
# Clase 0: máquina apagada → guardar como data/reposo.csv
# Clase 1: maquinado normal → guardar como data/operacion_normal.csv
# Clase 2: golpear/sacudir la máquina → guardar como data/anomalia.csv
```

Formato del CSV:
```
timestamp,accel_x,accel_y,accel_z,temperatura,humedad,label
1234567890,0.12,-0.03,9.81,27.4,62.1,0
```

---

## Entrenamiento del modelo (David)

```bash
# 1. Instalar dependencias
pip install tensorflow scikit-learn numpy pandas matplotlib joblib

# 2. Entrenar el modelo en Keras
python model/train_mlp.py

# 3. Exportar a TF Lite y generar archivos para el ESP32
python model/export_weights.py
# → Genera: firmware/inference/model.h
# → Genera: firmware/inference/scaler_params.h
```

El script de exportación genera:
- `model/model.tflite` — modelo TF Lite
- `firmware/inference/model.h` — array C++ listo para Arduino
- `firmware/inference/scaler_params.h` — parámetros de normalización para el ESP32

---

## Despliegue en ESP32 (Ktalyna)

```
1. Abrir Arduino IDE
2. Instalar librería: TFLite_ESP32 by Eloquent Arduino
3. Abrir firmware/inference/cnc_mlp_inference.ino
4. Verificar que model.h y scaler_params.h tienen los valores reales (generados por David)
5. Seleccionar placa: ESP32C3 Dev Module
6. Compilar y cargar
7. Abrir Serial Monitor a 115200 baud
```

Salida esperada en Serial Monitor:
```
=== FLUX CNC — TF Lite Micro Edge Inference ===
Arena usada: XXXX bytes
Listo. Recolectando ventana inicial...
──────────────────────────────
Temp: 27.40 C  Hum: 62.10%
Media   X:0.012 Y:-0.003 Z:9.810
Varianza X:0.0001 Y:0.0001 Z:0.0002
Prediccion: [1] OPERACION_NORMAL
Probs: R=0.02 ON=0.95 AN=0.03
```

---

## Dependencias Arduino

Instalar desde el Library Manager del Arduino IDE:
- **TFLite_ESP32** by Eloquent Arduino
- **DHT sensor library** by Adafruit
- **Wire** (incluida en ESP32 core)

---

## Comparación con entregas anteriores

| Aspecto | Entrega 2 (MQTT+InfluxDB) | Entrega 3 (Azure IoT) | Esta entrega (Edge AI) |
|---------|--------------------------|----------------------|------------------------|
| Dónde procesa | Servidor (nube) | Azure Functions | ESP32 (edge) |
| Latencia | ~100ms+ | ~500ms+ | < 1ms |
| Requiere internet | Sí | Sí | No |
| Tipo de análisis | Umbrales fijos | Umbrales fijos | Red neuronal MLP |
| Framework ML | — | — | TF Lite Micro |

---

## Criterios de evaluación

| Criterio | Cómo verificarlo |
|----------|-----------------|
| ✅ Dataset etiquetado | Archivos CSV en `data/` con 600-1000 muestras totales |
| ✅ Modelo entrenado | `model/mlp_model.keras` + gráficas Accuracy/Loss |
| ✅ Exportación TF Lite | `model/model.tflite` + `firmware/inference/model.h` generados |
| ✅ Inferencia en ESP32 | Serial Monitor mostrando predicciones en tiempo real |
| ✅ README con lógica | Este documento |