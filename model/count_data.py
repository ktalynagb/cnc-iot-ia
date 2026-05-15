"""Count raw samples per class from data/ CSVs."""
import json, glob, pandas as pd, os

DATA_DIR = "data"
OUT_PATH = "model/data_counts.json"

dfs = []
for f in sorted(glob.glob(f"{DATA_DIR}/*.csv")):
    df = pd.read_csv(f)
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
counts = all_data["label"].value_counts().sort_index()
n_min = int(counts.min())

result = {
    "per_class": {str(int(k)): int(v) for k, v in counts.items()},
    "total": int(len(all_data)),
    "n_min": n_min,
    "classes": {
        "0": "reposo",
        "1": "operacion_normal",
        "2": "anomalia"
    }
}

os.makedirs("model", exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(f"Conteos por clase: {result['per_class']}")
print(f"Total: {result['total']}")
print(f"n_min: {n_min}")
print(f"Guardado en {OUT_PATH}")
