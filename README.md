# 🏭 CNC IoT — Clasificador de Estados Vibracionales · Edge AI

Proyecto de la **Especialización en Inteligencia Artificial aplicada a IoT**  
Universidad Autónoma de Occidente · Práctica 4 — MLP en ESP32 (TinyML / Edge AI)

---

## Integrantes

| Rol | Persona | Responsabilidad |
|-----|---------|-----------------|
| Hardware & Recolección | Valentina | ESP32-C3 + MPU-6050 + DHT22 · Firmware de captura de datos etiquetados |
| Entrenamiento MLP | David | Dataset CSV · Entrenamiento Python · Export de pesos a C++ |
| Firmware & Despliegue | Ktalyna | Inferencia MLP en ESP32 · Forward pass en C++ · Integración Edge AI |

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
│         │    MLP: 8 → 16 → 3     │  TinyML      │
│         │   ReLU      Softmax     │  Edge AI     │
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
| 3. Entrenamiento MLP | David (Python/Colab) | sklearn MLPClassifier 8→16→3 |
| 4. Evaluación | David | Accuracy, Loss, matriz de confusión |
| 5. Exportación (Compilation) | David → `export_weights.py` | Pesos Python → arrays C++ |
| 6. Despliegue Edge | Ktalyna | Forward pass en ESP32, sin librerías externas |

---

## Hardware

| Componente | Conexión | Función |
|------------|----------|---------|
| ESP32-C3 Super Mini | — | Microcontrolador. Corre la inferencia MLP en tiempo real |
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
│       ├── cnc_mlp_inference.ino # Firmware inferencia Edge AI (Ktalyna)
│       ├── mlp_inference.h       # Forward pass MLP en C++ puro
│       └── model_weights.h       # Pesos del modelo (generado por export_weights.py)
├── data/
│   ├── reposo.csv                # Dataset clase 0 (David)
│   ├── operacion_normal.csv      # Dataset clase 1 (David)
│   └── anomalia.csv              # Dataset clase 2 (David)
├── model/
│   ├── train_mlp.py              # Script entrenamiento (David)
│   ├── export_weights.py         # Exportar pesos Python → C++ (David → Ktalyna)
│   ├── mlp_model.pkl             # Modelo entrenado serializado
│   └── scaler.pkl                # Parámetros de normalización
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
pip install scikit-learn numpy pandas matplotlib joblib

# 2. Entrenar el modelo
python model/train_mlp.py

# 3. Exportar pesos a C++
python model/export_weights.py
# → Genera: firmware/inference/model_weights.h
```

El script de entrenamiento genera:
- `model/mlp_model.pkl` — modelo serializado
- `model/scaler.pkl` — parámetros de normalización
- Gráficas de Accuracy/Loss
- `firmware/inference/model_weights.h` — listo para el ESP32

---

## Despliegue en ESP32 (Ktalyna)

```
1. Abrir Arduino IDE
2. Abrir firmware/inference/cnc_mlp_inference.ino
3. Verificar que model_weights.h tiene los pesos reales (no placeholders)
4. Seleccionar placa: ESP32C3 Dev Module
5. Compilar y cargar
6. Abrir Serial Monitor a 115200 baud
```

Salida esperada en Serial Monitor:
```
=== FLUX CNC — MLP Edge Inference ===
──────────────────────────────
Temp: 27.40°C  Hum: 62.10%
Accel media  X:0.012 Y:-0.003 Z:9.810
Accel var    X:0.0001 Y:0.0001 Z:0.0002
Prediccion: [1] OPERACION_NORMAL
Probs: R=0.02 ON=0.95 AN=0.03
```

---

## Dependencias Arduino

Instalar desde el Library Manager del Arduino IDE:
- **DHT sensor library** by Adafruit
- **Wire** (incluida en ESP32 core)

> No se requiere TensorFlow Lite ni librerías de ML — el forward pass está implementado en C++ puro en `mlp_inference.h`.

---

## Comparación con entregas anteriores

| Aspecto | Entrega 2 (MQTT+InfluxDB) | Entrega 3 (Azure IoT) | Esta entrega (Edge AI) |
|---------|--------------------------|----------------------|------------------------|
| Dónde procesa | Servidor (nube) | Azure Functions | ESP32 (edge) |
| Latencia | ~100ms+ | ~500ms+ | < 1ms |
| Requiere internet | Sí | Sí | No |
| Tipo de análisis | Umbrales fijos | Umbrales fijos | Red neuronal MLP |
| Escala | Multi-dispositivo | Multi-dispositivo | Un dispositivo |

---

## Criterios de evaluación

| Criterio | Cómo verificarlo |
|----------|-----------------|
| ✅ Dataset etiquetado | Archivos CSV en `data/` con 600-1000 muestras totales |
| ✅ Modelo entrenado | `model/mlp_model.pkl` + gráficas Accuracy/Loss |
| ✅ Pesos exportados | `firmware/inference/model_weights.h` generado por `export_weights.py` |
| ✅ Inferencia en ESP32 | Serial Monitor mostrando predicciones en tiempo real |
| ✅ README con lógica | Este documento |
