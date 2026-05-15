# Makefile — FLUX CNC IoT · MLP Pipeline
# Reproducible pipeline for training, exporting and validating the MLP model.
# Uses a Python virtual environment named "uv".

PYTHON    := uv/bin/python
UV_DIR    := uv
SEED      := 42

.PHONY: env count preprocess train export validate readme all clean

## Create virtual environment and install dependencies
env:
	python -m venv $(UV_DIR)
	$(UV_DIR)/bin/pip install --upgrade pip
	$(UV_DIR)/bin/pip install tensorflow scikit-learn numpy pandas plotly matplotlib joblib kaleido
	$(UV_DIR)/bin/pip freeze > model/requirements.txt
	@echo "Entorno 'uv' listo. Activa con: source uv/bin/activate"

## Count raw samples per class → model/data_counts.json
count:
	$(PYTHON) model/count_data.py

## Balance, window features, split → model/train.csv val.csv test.csv + scaler.pkl
preprocess:
	$(PYTHON) model/preprocess.py

## Train MLP → model/mlp_model.keras + model/training_history.json
train:
	$(PYTHON) model/train_mlp.py

## Export to TFLite + model.h + scaler_params.h
export:
	$(PYTHON) model/export_weights.py

## Validate: metrics, Plotly figures, report.txt
validate:
	$(PYTHON) model/validate.py

## Run full pipeline
all: count preprocess train export validate

## Update README (placeholder - already done manually)
readme:
	@echo "README actualizado manualmente"

## Create PR
pr:
	@echo "Crear PR manualmente o via GitHub CLI: gh pr create"

clean:
	rm -rf model/train.csv model/val.csv model/test.csv
	rm -rf model/dataset_balanced.csv model/mlp_model.keras
	rm -rf model/model.tflite model/scaler.pkl model/training_history.json
	rm -rf model/figures/ model/report.txt model/data_counts.json
