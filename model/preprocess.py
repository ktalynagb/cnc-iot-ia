"""Preprocess: window features, balance, split, scaler."""
import numpy as np
import pandas as pd
import joblib, glob, os, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
WINDOW = 32
DATA_DIR = "data"
MODEL_DIR = "model"
SCALER_H = "firmware/inference/scaler_params.h"

np.random.seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SCALER_H), exist_ok=True)

# Load raw data per class
class_dfs = {}
for f in sorted(glob.glob(f"{DATA_DIR}/*.csv")):
    df = pd.read_csv(f)
    for label in df["label"].unique():
        sub = df[df["label"] == label].reset_index(drop=True)
        class_dfs[int(label)] = class_dfs.get(int(label), pd.DataFrame())
        class_dfs[int(label)] = pd.concat([class_dfs[int(label)], sub], ignore_index=True)

# Extract features per class using non-overlapping windows
def extract_windows(df, window=32):
    rows = []
    n = len(df)
    for i in range(0, n - window + 1, window):
        w = df.iloc[i:i+window]
        feat = [
            w["accel_x"].mean(), w["accel_x"].var(),
            w["accel_y"].mean(), w["accel_y"].var(),
            w["accel_z"].mean(), w["accel_z"].var(),
            w["temperatura"].mean(), w["humedad"].mean(),
        ]
        label = int(w["label"].iloc[0])
        rows.append(feat + [label])
    cols = ["mean_x","var_x","mean_y","var_y","mean_z","var_z","temperature","humidity","label"]
    return pd.DataFrame(rows, columns=cols)

windows_per_class = {}
for label, df in sorted(class_dfs.items()):
    wdf = extract_windows(df)
    windows_per_class[label] = wdf
    print(f"Clase {label}: {len(df)} muestras raw -> {len(wdf)} ventanas")

# Balance by downsampling to n_min windows
n_min = min(len(v) for v in windows_per_class.values())
print(f"n_min (ventanas): {n_min}")

balanced_parts = []
for label in sorted(windows_per_class.keys()):
    df = windows_per_class[label]
    if len(df) > n_min:
        df = df.sample(n=n_min, random_state=SEED).reset_index(drop=True)
    balanced_parts.append(df)

balanced = pd.concat(balanced_parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
balanced.to_csv(f"{MODEL_DIR}/dataset_balanced.csv", index=False)
print(f"Dataset balanceado: {len(balanced)} ventanas ({n_min} x {len(windows_per_class)} clases)")

# Update data_counts.json with window counts
counts_path = f"{MODEL_DIR}/data_counts.json"
if os.path.exists(counts_path):
    with open(counts_path) as f:
        counts_data = json.load(f)
else:
    counts_data = {}
counts_data["n_min_windows"] = n_min
counts_data["windows_per_class"] = {str(k): len(v) for k, v in windows_per_class.items()}
counts_data["total_balanced_windows"] = len(balanced)
with open(counts_path, "w") as f:
    json.dump(counts_data, f, indent=2)

# Stratified split 70/20/10
X = balanced.drop("label", axis=1).values
y = balanced["label"].values
feat_cols = list(balanced.drop("label", axis=1).columns)

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.10, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.20/0.90, random_state=SEED, stratify=y_tmp)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Fit scaler on train only
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
print(f"Scaler guardado en {MODEL_DIR}/scaler.pkl")

# Save CSVs (scaled features + label)
def save_csv(X, y, path):
    df = pd.DataFrame(X, columns=feat_cols)
    df["label"] = y
    df.to_csv(path, index=False)

save_csv(X_train_sc, y_train, f"{MODEL_DIR}/train.csv")
save_csv(X_val_sc,   y_val,   f"{MODEL_DIR}/val.csv")
save_csv(X_test_sc,  y_test,  f"{MODEL_DIR}/test.csv")
print("train/val/test.csv guardados")

# Generate scaler_params.h
mean = scaler.mean_
std  = scaler.scale_

def fmt_array(arr, name):
    vals = ", ".join(f"{v:.8f}f" for v in arr)
    return f"const float {name}[{len(arr)}] = {{\n  {vals}\n}};\n"

header = f"""/**
 * scaler_params.h - FLUX CNC IoT - Edge AI
 * GENERADO AUTOMATICAMENTE por model/preprocess.py
 * NO EDITAR MANUALMENTE
 *
 * Features (orden):
 *   [0] media(accel_x)   [1] varianza(accel_x)
 *   [2] media(accel_y)   [3] varianza(accel_y)
 *   [4] media(accel_z)   [5] varianza(accel_z)
 *   [6] temperatura      [7] humedad
 */

#pragma once

{fmt_array(mean, "SCALER_MEAN")}
{fmt_array(std, "SCALER_STD")}
"""

with open(SCALER_H, "w") as f:
    f.write(header)
print(f"scaler_params.h generado en {SCALER_H}")
