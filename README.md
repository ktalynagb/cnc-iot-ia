# CNC IoT — Clasificador de Estados Vibracionales · Edge AI

Proyecto de la **Especialización en Inteligencia Artificial aplicada a IoT**  
Universidad Autónoma de Occidente · Práctica 4 — MLP en ESP32 (TinyML / Edge AI)

---

## Integrantes

| Rol | Persona | Responsabilidad |
|-----|---------|-----------------|
| Hardware & Recolección | Valentina | ESP32-C3 + MPU-6050 + DHT11 · Firmware de captura de datos etiquetados |
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
│  DHT11    ──► Temperatura · Humedad              │
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
| DHT11 | GPIO0 | Sensor temperatura (°C) y humedad (%) |
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
| 6 | temperatura | Valor directo DHT11 (°C) |
| 7 | humedad | Valor directo DHT11 (%) |

> La temperatura y humedad se usan como valor directo porque el DHT11 mide lento (cada 2s) — calcular varianza no aporta información útil para detección de vibración.

---

## Estructura del repositorio

```
cnc-iot-ia/
├── Esp32/
│   ├── captura_datos.py          # Script Python — guarda CSV etiquetados desde Serial (Valentina)
│   └── cnc_iot_esp32/
│       ├── cnc_iot_esp32.ino     # Firmware de captura: MPU-6050 + DHT11 → Serial CSV (Valentina)
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

El firmware `cnc_iot_esp32.ino` transmite lecturas del MPU-6050 y DHT11 por Serial a ~100 Hz.
El script `captura_datos.py` las recibe y las guarda en CSV etiquetados listos para el entrenamiento.

**1. Subir el firmware al ESP32**

```
Arduino IDE → Abrir Esp32/cnc_iot_esp32/cnc_iot_esp32.ino
Placa: ESP32C3 Dev Module
Compilar y cargar
```

**2. Instalar dependencias Python (una sola vez)**

```bash
pip install pyserial
```

**3. Correr el script de captura**

```bash
python Esp32/captura_datos.py
```

El script detecta el puerto automáticamente, pregunta qué clase capturar y muestra una barra de progreso. Capturar **200-350 muestras por clase** en cada estado:

| Clase | Estado | Cómo simularlo |
|-------|--------|----------------|
| 0 | Reposo | ESP32 quieto, máquina apagada |
| 1 | Operación Normal | ESP32 sobre máquina en funcionamiento normal |
| 2 | Anomalía | Golpear/sacudir el ESP32 fuertemente |

Formato del CSV generado:
```
timestamp_ms,accel_x,accel_y,accel_z,temperatura,humedad,label
1234567890,0.1200,-0.0300,9.8100,27.40,62.10,0
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
- **TFLite_ESP32** by Eloquent Arduino — para inferencia (Ktalyna)
- **DHT sensor library** by Adafruit — para DHT11 (ambos firmwares)
- **Wire** (incluida en ESP32 core)

Dependencia Python (recolección de datos):
```bash
pip install pyserial
```

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

---

## Model training & export (David)

All model steps are reproducible via `make` using the `uv` virtual environment.

### Crear entorno virtual e instalar dependencias

```bash
make env
# O manualmente:
python -m venv uv
source uv/bin/activate          # Linux/macOS
# uv\Scripts\activate           # Windows
pip install tensorflow scikit-learn numpy pandas plotly joblib kaleido
```

### Pipeline completo

```bash
# 1. Contar muestras por clase
make count
# → model/data_counts.json

# 2. Preprocesar: balanceo, features, splits, scaler
make preprocess
# → model/train.csv, model/val.csv, model/test.csv
# → model/scaler.pkl, firmware/inference/scaler_params.h

# 3. Entrenar el modelo MLP
make train
# → model/mlp_model.keras, model/training_history.json

# 4. Exportar para firmware
make export
# → model/model.tflite, firmware/inference/model.h

# 5. Validar y generar reportes
make validate
# → model/report.txt, model/figures/

# O ejecutar todo en un paso:
make all
```

### Arquitectura del modelo

| Capa | Tipo | Activación |
|------|------|-----------|
| Input | Dense | — |
| Oculta | Dense(16) | ReLU |
| Salida | Dense(3) | Softmax |

- Épocas: 32 · Batch: 32 · Adam(lr=0.01) · Seed: 42
- Balanceo: downsampling a n_min ventanas por clase (seed=42)
- Split estratificado: 70% train / 20% val / 10% test

### Resultados (seed=42, 32 épocas)

| Métrica | Keras | TFLite |
|---------|-------|--------|
| Test Accuracy | **0.9149** | **0.9149** |
| Max diff probabilidades | — | 2.2e-07 |

### Artefactos generados

| Archivo | Descripción |
|---------|-------------|
| `model/mlp_model.keras` | Modelo Keras (mejor val_loss) |
| `model/model.tflite` | Modelo TFLite float32 |
| `model/scaler.pkl` | StandardScaler serializado |
| `firmware/inference/model.h` | Array C++ para ESP32 |
| `firmware/inference/scaler_params.h` | Parámetros Z-score para ESP32 |
| `model/report.txt` | Métricas y validación |
| `model/figures/` | Gráficas Plotly (HTML + PNG) |