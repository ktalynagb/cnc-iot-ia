"""Train MLP classifier for CNC vibration states."""
import os, json, numpy as np, pandas as pd
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR = "model"
EPOCHS = 32
BATCH  = 32
LR     = 0.01

# GPU detection
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs disponibles: {gpus}")
device = "/GPU:0" if gpus else "/CPU:0"
print(f"Usando dispositivo: {device}")

# Load data
def load_csv(path):
    df = pd.read_csv(path)
    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    return X, y

X_train, y_train = load_csv(f"{MODEL_DIR}/train.csv")
X_val,   y_val   = load_csv(f"{MODEL_DIR}/val.csv")
X_test,  y_test  = load_csv(f"{MODEL_DIR}/test.csv")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Build model
with tf.device(device):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3,  activation="softmax"),
    ], name="mlp_cnc")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"{MODEL_DIR}/mlp_model.keras",
        monitor="val_loss", save_best_only=True, verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[checkpoint],
        verbose=2,
    )

# Save training history
hist_dict = {k: [float(v) for v in vs] for k, vs in history.history.items()}
with open(f"{MODEL_DIR}/training_history.json", "w") as f:
    json.dump(hist_dict, f, indent=2)
print(f"Historial guardado en {MODEL_DIR}/training_history.json")

# Quick eval on test
model_best = tf.keras.models.load_model(f"{MODEL_DIR}/mlp_model.keras")
loss, acc = model_best.evaluate(X_test, y_test, verbose=0)
print(f"Test loss={loss:.4f}  acc={acc:.4f}")
