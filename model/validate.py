"""Validate model: metrics, Plotly figures, TFLite inference check, report."""
import os, json, numpy as np, pandas as pd, joblib
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import plotly.graph_objects as go
import plotly.express as px

SEED = 42
MODEL_DIR   = "model"
FIGURES_DIR = f"{MODEL_DIR}/figures"
TFLITE_PATH = f"{MODEL_DIR}/model.tflite"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load history
with open(f"{MODEL_DIR}/training_history.json") as f:
    hist = json.load(f)

# Load data and scaler
def load_csv(path):
    df = pd.read_csv(path)
    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    return X, y

X_test, y_test = load_csv(f"{MODEL_DIR}/test.csv")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

# Keras model predictions
model = tf.keras.models.load_model(f"{MODEL_DIR}/mlp_model.keras")
y_pred_keras = np.argmax(model.predict(X_test, verbose=0), axis=1)
acc_keras = accuracy_score(y_test, y_pred_keras)
print(f"Keras acc: {acc_keras:.4f}")

# --- Plotly: Loss curve ---
epochs = list(range(1, len(hist["loss"]) + 1))
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=epochs, y=hist["loss"],     mode="lines+markers", name="Train Loss"))
fig_loss.add_trace(go.Scatter(x=epochs, y=hist["val_loss"], mode="lines+markers", name="Val Loss"))
fig_loss.update_layout(title="Loss vs Epoch", xaxis_title="Epoch", yaxis_title="Loss",
                       template="plotly_white")
fig_loss.write_html(f"{FIGURES_DIR}/loss_curve.html")
fig_loss.write_image(f"{FIGURES_DIR}/loss_curve.png", engine="kaleido")
print("loss_curve guardado")

# --- Plotly: Accuracy curve ---
fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(x=epochs, y=hist["accuracy"],     mode="lines+markers", name="Train Acc"))
fig_acc.add_trace(go.Scatter(x=epochs, y=hist["val_accuracy"], mode="lines+markers", name="Val Acc"))
fig_acc.update_layout(title="Accuracy vs Epoch", xaxis_title="Epoch", yaxis_title="Accuracy",
                      template="plotly_white")
fig_acc.write_html(f"{FIGURES_DIR}/accuracy_curve.html")
fig_acc.write_image(f"{FIGURES_DIR}/accuracy_curve.png", engine="kaleido")
print("accuracy_curve guardado")

# --- Confusion matrix ---
LABELS = ["Reposo", "Op.Normal", "Anomalía"]
cm = confusion_matrix(y_test, y_pred_keras)
fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                   x=LABELS, y=LABELS,
                   labels=dict(x="Predicho", y="Real", color="Cuenta"),
                   title="Matriz de Confusión (Test Set)")
fig_cm.write_html(f"{FIGURES_DIR}/confusion_matrix.html")
fig_cm.write_image(f"{FIGURES_DIR}/confusion_matrix.png", engine="kaleido")
print("confusion_matrix guardado")

# --- TFLite validation ---
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp_det = interp.get_input_details()
out_det = interp.get_output_details()

y_pred_tflite = []
for row in X_test:
    interp.set_tensor(inp_det[0]["index"], row.reshape(1, -1))
    interp.invoke()
    out = interp.get_tensor(out_det[0]["index"])
    y_pred_tflite.append(np.argmax(out))
y_pred_tflite = np.array(y_pred_tflite)
acc_tflite = accuracy_score(y_test, y_pred_tflite)
print(f"TFLite acc: {acc_tflite:.4f}")

# Check max diff between keras and tflite predictions (raw probabilities)
max_diff = 0.0
for row in X_test[:100]:
    keras_out = model.predict(row.reshape(1,-1), verbose=0)[0]
    interp.set_tensor(inp_det[0]["index"], row.reshape(1,-1))
    interp.invoke()
    tflite_out = interp.get_tensor(out_det[0]["index"])[0]
    diff = np.max(np.abs(keras_out - tflite_out))
    if diff > max_diff:
        max_diff = diff

print(f"Max diff Keras vs TFLite (probabilities): {max_diff:.8f}")

# Save TFLite predictions
pred_df = pd.DataFrame(X_test, columns=["mean_x","var_x","mean_y","var_y","mean_z","var_z","temperature","humidity"])
pred_df["label_true"] = y_test
pred_df["label_pred_tflite"] = y_pred_tflite
pred_df.to_csv(f"{MODEL_DIR}/test_predictions_tflite.csv", index=False)
print(f"test_predictions_tflite.csv guardado")

# Check scaler_params.h consistency
scaler_h_path = "firmware/inference/scaler_params.h"
with open(scaler_h_path) as f:
    content = f.read()

import re
mean_match = re.search(r"SCALER_MEAN\[\d+\]\s*=\s*\{([^}]+)\}", content)
std_match  = re.search(r"SCALER_STD\[\d+\]\s*=\s*\{([^}]+)\}", content)
h_mean = np.array([float(x.strip().rstrip("f")) for x in mean_match.group(1).split(",")])
h_std  = np.array([float(x.strip().rstrip("f")) for x in std_match.group(1).split(",")])
scaler_mean_diff = np.max(np.abs(h_mean - scaler.mean_))
scaler_std_diff  = np.max(np.abs(h_std  - scaler.scale_))
print(f"Max diff scaler mean (h vs pkl): {scaler_mean_diff:.2e}")
print(f"Max diff scaler std  (h vs pkl): {scaler_std_diff:.2e}")

# Classification report
report_str = classification_report(y_test, y_pred_keras, target_names=LABELS)
print(report_str)

# Load counts
with open(f"{MODEL_DIR}/data_counts.json") as f:
    counts_data = json.load(f)

# Load split sizes for report
import pandas as _pd
_tr = _pd.read_csv(f"{MODEL_DIR}/train.csv")
_va = _pd.read_csv(f"{MODEL_DIR}/val.csv")
_te = _pd.read_csv(f"{MODEL_DIR}/test.csv")

# Write report.txt
with open(f"{MODEL_DIR}/report.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("FLUX CNC IoT - MLP Pipeline Report\n")
    f.write("="*60 + "\n\n")
    f.write("=== Conteo de datos raw ===\n")
    for cls, cnt in counts_data["per_class"].items():
        name = counts_data["classes"][cls]
        f.write(f"  Clase {cls} ({name}): {cnt} muestras\n")
    f.write(f"  Total raw: {counts_data['total']}\n\n")
    f.write("=== Ventanas extraídas (window=32, non-overlapping) ===\n")
    for cls, cnt in counts_data["windows_per_class"].items():
        f.write(f"  Clase {cls}: {cnt} ventanas\n")
    f.write(f"  n_min (ventanas): {counts_data['n_min_windows']}\n")
    f.write(f"  Total balanceado: {counts_data['total_balanced_windows']} ventanas\n\n")
    f.write("=== Splits (70/20/10 estratificado, seed=42) ===\n")
    f.write(f"  Train: {len(_tr)}, Val: {len(_va)}, Test: {len(_te)}\n\n")
    f.write("=== Scaler (StandardScaler, fit solo en train) ===\n")
    f.write(f"  Mean: {scaler.mean_.tolist()}\n")
    f.write(f"  Std:  {scaler.scale_.tolist()}\n")
    f.write(f"  Diff scaler_params.h vs scaler.pkl:\n")
    f.write(f"    mean: max_diff={scaler_mean_diff:.2e}\n")
    f.write(f"    std:  max_diff={scaler_std_diff:.2e}\n\n")
    f.write("=== Métricas modelo (test set) ===\n")
    f.write(f"  Keras acc:  {acc_keras:.4f}\n")
    f.write(f"  TFLite acc: {acc_tflite:.4f}\n")
    f.write(f"  Max prob diff Keras vs TFLite: {max_diff:.8f}\n\n")
    f.write("=== Classification Report (Keras) ===\n")
    f.write(report_str + "\n")
    f.write("=== Matriz de Confusión (Keras, test) ===\n")
    f.write(str(cm) + "\n\n")
    f.write("=== Figuras generadas ===\n")
    f.write("  model/figures/loss_curve.html + .png\n")
    f.write("  model/figures/accuracy_curve.html + .png\n")
    f.write("  model/figures/confusion_matrix.html + .png\n\n")
    f.write("=== Artefactos generados ===\n")
    f.write("  model/mlp_model.keras\n")
    f.write("  model/model.tflite\n")
    f.write("  model/scaler.pkl\n")
    f.write("  firmware/inference/model.h\n")
    f.write("  firmware/inference/scaler_params.h\n\n")
    f.write("=== Pasos siguientes (deploy en ESP32) ===\n")
    f.write("  1. Abrir Arduino IDE\n")
    f.write("  2. Instalar TFLite_ESP32 by Eloquent Arduino\n")
    f.write("  3. Abrir firmware/inference/cnc_mlp_inference.ino\n")
    f.write("  4. Verificar que model.h y scaler_params.h están en el mismo directorio\n")
    f.write("  5. Compilar y cargar en ESP32-C3\n")
    f.write("  6. Abrir Serial Monitor a 115200 baud\n")

print(f"report.txt guardado en {MODEL_DIR}/report.txt")
