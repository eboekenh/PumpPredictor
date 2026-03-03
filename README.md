# 💧 Pump It Up: Data Mining the Water Table

> **End-to-end Machine Learning project for predictive maintenance of water pumps in Tanzania — built for social impact.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📖 About the Project

This project is based on the [DrivenData "Pump It Up" competition](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/), which focuses on predicting the operational status of water pumps in Tanzania.

Understanding and optimizing water resource infrastructure is critical for public health and development across Sub-Saharan Africa. This project demonstrates how data science can be applied to real-world social impact challenges.

---

## 🎯 Problem Statement

Build a multi-class classifier to predict whether a given water pump is:

| Label | Description |
|---|---|
| ✅ `functional` | Pump is operational |
| ⚠️ `functional needs repair` | Pump works but requires maintenance |
| ❌ `non functional` | Pump is not working |

---

## ✨ Features

- **End-to-end ML pipeline**: from raw data ingestion to packaged model
- **Multi-class classification** on real-world geospatial & categorical data
- **Reproducible structure** with `setup.py` for installable packaging
- **Mentoring-ready**: designed as a teaching project for ReDI School students

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8 | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | ML modeling & evaluation |
| Matplotlib / Seaborn | Visualization |
| Git & GitHub | Version control |
| VS Code + Anaconda | Development environment |

---

## 📦 Project Structure

```
PumpPredictor/
├── README.md
├── setup.py                         # Installable package config
├── requirements.txt
├── .gitignore
├── artifacts/                       # Model artifacts (not committed)
│   └── model.joblib
├── scripts/
│   ├── train.py                     # CLI: train a classifier
│   └── predict.py                   # CLI: generate predictions
├── src/
│   └── pumpitup/
│       ├── config.py                # Constants & default paths
│       ├── data/
│       │   ├── io.py                # CSV loading & saving
│       │   └── synthetic.py        # Synthetic dataset generator
│       ├── features/
│       │   └── preprocess.py       # ColumnTransformer pipeline
│       ├── models/
│       │   ├── train.py             # train_model, save_model
│       │   └── predict.py          # load_model, predict
│       └── evaluation/
│           └── metrics.py          # accuracy, f1_macro
└── tests/
    └── test_smoke_pipeline.py       # Smoke tests
```

> **Note:** The DrivenData dataset is **not committed** to this repository.
> Download it from [DrivenData](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/)
> and supply paths via CLI arguments (see below).

---

## 🗺️ Roadmap

- [x] Project setup & reproducible structure
- [x] Modular package under `src/pumpitup/`
- [x] Synthetic data generator for local smoke tests
- [x] End-to-end training + inference pipeline
- [x] CLI entrypoints (`scripts/train.py`, `scripts/predict.py`)
- [ ] Hyperparameter tuning
- [ ] Deployment-ready packaging

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/eboekenh/PumpPredictor.git
cd PumpPredictor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies (includes the pumpitup package in editable mode)
pip install -r requirements.txt
```

---

## 🏋️ Training

### On synthetic data (no real dataset required)

```bash
python scripts/train.py
```

This generates 500 synthetic samples, trains a Random Forest classifier, prints
in-sample metrics, and saves the model to `artifacts/model.joblib`.

### On real DrivenData CSVs

```bash
python scripts/train.py \
    --train-csv data/train_values.csv \
    --target status_group \
    --model-output artifacts/model.joblib
```

---

## 🔮 Predicting

```bash
python scripts/predict.py \
    --model artifacts/model.joblib \
    --input data/test_values.csv \
    --output artifacts/predictions.csv
```

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License.

---

