# 💧 Pump It Up — Predicting Water Pump Failures in Tanzania

End-to-end machine learning pipeline for the [DrivenData "Pump It Up"](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) competition. Multi-class classification to predict the operational status of water pumps across Tanzania — a real-world social impact problem affecting millions of people's access to clean water.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-brightgreen.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Problem

Predict whether a water pump is **functional**, **needs repair**, or **non-functional** based on ~40 features including location, construction details, water source, and management information. The dataset contains **59,400 water points** across Tanzania.

| Label | Description | Distribution |
|-------|-------------|-------------|
| ✅ `functional` | Pump is operational | ~54% |
| ⚠️ `functional needs repair` | Works but needs maintenance | ~7% |
| ❌ `non functional` | Pump is broken | ~39% |

---

## Results

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Random Forest | ~80% | ~0.73 | `n_estimators=100`, `max_depth=20`, `class_weight='balanced'` |
| XGBoost | ~81% | ~0.75 | `n_estimators=200`, `max_depth=8`, early stopping |
| **LightGBM** | **~82%** | **~0.76** | `n_estimators=500`, `learning_rate=0.05`, `num_leaves=63` |

Results on validation set (80/20 stratified split).

---

## Pipeline

```
Raw Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction
```

### 1. Exploratory Data Analysis
- Target distribution analysis (class imbalance: 7:1 ratio for `needs repair`)
- Categorical variable analysis with cross-tabulation against target
- Numerical variable distributions with box plots by pump status
- Geographical visualization of 59K water points across Tanzania

### 2. Preprocessing
- **Missing values:** Median imputation for numerical, mode for categorical, special handling for zero-encoded missing values (construction_year, gps_height, population)
- **Encoding:** LabelEncoder fitted on combined train+test to ensure consistent mappings
- **Scaling:** StandardScaler on numerical features (fitted on train only)

### 3. Feature Engineering  (`src/pumpitup/features/engineering.py`)
- **Date features:** Pump age, age category, recording month/season
- **Geographical features:** Haversine distance from Tanzania center, elevation categories
- **Aggregation features:** Per-region/basin/installer statistics (pump count, avg age, failure rate)
  — aggregations are derived **only** from training data to avoid leakage

### 4. Models
Three classifiers compared with consistent evaluation:
- **Random Forest** — Balanced class weights, strong baseline
- **XGBoost** — Gradient boosting with early stopping
- **LightGBM** — Best performance, fastest training

---

## Project Structure

```
PumpPredictor/
├── pump_analysis_full_pipeline.py   # Full analysis: EDA → models → submission (standalone)
├── notebooks/
│   └── 01_eda_analysis.ipynb        # Interactive EDA notebook
├── src/pumpitup/                    # Modular, installable Python package
│   ├── config.py                    # Constants & default paths
│   ├── data/
│   │   ├── io.py                    # CSV loading & saving
│   │   └── synthetic.py             # Synthetic dataset generator
│   ├── features/
│   │   ├── preprocess.py            # ColumnTransformer pipeline
│   │   └── engineering.py           # Feature engineering (age, geo, aggregations)
│   ├── models/
│   │   ├── train.py                 # Model training + feature importance
│   │   └── predict.py               # Inference
│   ├── evaluation/
│   │   └── metrics.py               # Accuracy, F1, cross-validation, model comparison
│   └── visualization/
│       └── plots.py                 # EDA & model evaluation plots
├── scripts/
│   ├── train.py                     # CLI: train a classifier
│   └── predict.py                   # CLI: generate predictions
├── tests/
│   ├── test_smoke_pipeline.py           # Smoke tests (pytest)
│   ├── test_feature_engineering.py      # Feature engineering unit tests
│   └── test_visualization_and_evaluation.py  # Visualization & evaluation tests
├── data/                            # Place competition CSVs here
├── artifacts/                       # Saved models (not committed)
├── requirements.txt
└── setup.py
```

**Two ways to use this project:**

| Approach | File | Best for |
|----------|------|----------|
| Standalone script | `pump_analysis_full_pipeline.py` | Full analysis walkthrough, learning |
| Modular package | `src/pumpitup/` + `scripts/` | Production use, testing, extending |

---

## Getting Started

```bash
git clone https://github.com/eboekenh/PumpPredictor.git
cd PumpPredictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### With real data
Download from [DrivenData](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/) and place CSVs in `data/`:

```bash
python scripts/train.py --train-csv data/training_set_values.csv --target status_group
```

### Quick test (no data download needed)
```bash
python scripts/train.py          # Uses synthetic data
pytest tests/                    # Run smoke tests
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Scikit-learn | ML pipelines, Random Forest, metrics |
| XGBoost | Gradient boosting classifier |
| LightGBM | Gradient boosting classifier |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| pytest | Testing |

---

## License

MIT
