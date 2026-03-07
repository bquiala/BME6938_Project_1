# CKD Risk Predictor — Machine Learning Application

> A reproducible Python machine learning application that predicts **Chronic Kidney Disease (CKD) risk** from structured tabular clinical data using a multi-model pipeline with an interactive Streamlit interface.

---

## Table of Contents

1. [Clinical Context](#clinical-context)
2. [Quick Start](#quick-start)
3. [Usage Guide](#usage-guide)
4. [Data Description](#data-description)
5. [Results Summary](#results-summary)
6. [Project Structure](#project-structure)
7. [Authors and Contributions](#authors-and-contributions)
8. [Dependencies](#dependencies)

---

## Clinical Context

**Who is this for?**
Clinicians, healthcare researchers, and students studying clinical data science.

**What clinical problem does it address?**
Chronic Kidney Disease (CKD) affects roughly **10 % of the global population** and is often diagnosed late because individual biomarkers — most notably serum creatinine — can remain within normal reference ranges during early disease stages.

This application analyses **up to 24 clinical features simultaneously**, including haematological parameters (hemoglobin, packed cell volume, red/white blood cell counts), biochemical markers (albumin, blood urea, serum creatinine, blood glucose, sodium, potassium), urinalysis findings (specific gravity, albumin, sugar, red blood cells, pus cells, bacteria), and patient history (hypertension, diabetes mellitus, coronary artery disease, appetite, pedal edema, anemia).

By combining these features through a trained ensemble model, the system can detect CKD patterns that no single biomarker reveals in isolation.

---

## Quick Start

### Prerequisites

- **Python 3.10 or later**
- `pip`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/BME6938_Project_1.git
cd BME6938_Project_1/ckd_ml_project

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Trains all models, evaluates them, and saves the best model artefact.
# Place your chronic_kidney_disease.arff in data/ first.
python run_pipeline.py

# Optional: specify a custom dataset path
python run_pipeline.py --data path/to/your_dataset.arff
```

Expected runtime: **3–10 minutes** on a modern laptop (varies with hardware and
whether GridSearchCV parallelism is available).

### Launch the Streamlit Application

```bash
streamlit run app/app.py
```

Then open **http://localhost:8501** in your browser.

**Computational requirements:** any modern CPU; no GPU required.
8 GB RAM recommended for the full GridSearchCV grid.

---

## Usage Guide

### Step 1 — Load the Dataset

**Via the Streamlit UI:**
1. Open the **📂 Dataset** tab.
2. Click *Upload .arff file* and select `chronic_kidney_disease.arff`.
3. Alternatively, click *Load from data/ directory* if you have already placed the file in `data/`.
4. Inspect the data preview, missing-value chart, and descriptive statistics.

**Via the CLI:**
```bash
python run_pipeline.py --data data/chronic_kidney_disease.arff
```

### Step 2 — Run Training

**Via the Streamlit UI:**
1. Open the **🔬 Training** tab.
2. Click **🚀 Run Full Pipeline**.
3. A progress bar tracks each stage: preprocessing → model training → evaluation → saving.
4. The Model Comparison table appears when complete.

**Via the CLI:**
```bash
python run_pipeline.py   # runs all steps automatically
```

### Step 3 — View Results and Visualisations

**Via the Streamlit UI:**
Open the **📊 Visualisations** tab to see:
- Class distribution bar chart
- Feature correlation heatmap
- Top-15 feature importances (Random Forest)
- Overlaid ROC curves for all four models
- Confusion matrix for the best model

**Via the CLI / filesystem:**
PNG files are saved to `logs/` after `run_pipeline.py` completes:
- `class_distribution.png`
- `correlation_heatmap.png`
- `feature_importances.png`
- `roc_curves.png`
- `confusion_matrix_best.png`

### Step 4 — Predict Individual Patient Risk

**Via the Streamlit UI:**
1. Open the **🩺 Predict** tab.
2. Fill in the numeric biomarker fields (or leave at defaults for typical values).
3. Select categorical observations from the dropdown menus.
4. Click **🔍 Predict CKD Risk**.

**Expected output:**
```
CKD Probability : 87.3%
Prediction      : 🔴 CKD
Risk Level      : High
```

Risk levels:
| Probability | Risk Level |
|---|---|
| ≥ 70 % | High |
| 40 – 69 % | Moderate |
| < 40 % | Low |

> **Disclaimer:** This tool is for research and educational purposes only. It does not constitute medical advice.

---

## Data Description

### Source

**UCI Machine Learning Repository — Chronic Kidney Disease Dataset**
- Original collector: Dr L. Jerlin Rubini, Alagappa University, India (2015)
- OpenML dataset ID: 40536
- URL: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease

### Format and Structure

The dataset is distributed in **ARFF (Attribute-Relation File Format)**, the OpenML standard, and is loaded via `scipy.io.arff`.

| Property | Value |
|---|---|
| Samples | 400 |
| Features | 24 (mixed numeric and categorical) |
| Target | `class` — `ckd` / `notckd` |
| Missing values | Present in most columns (handled by KNN imputation) |

### Key Features

| Feature | Type | Clinical Relevance |
|---|---|---|
| `hemo` (hemoglobin) | Numeric | Strong CKD predictor |
| `sg` (specific gravity) | Numeric | Renal concentrating ability |
| `al` (albumin) | Numeric | Indicator of proteinuria |
| `bgr` (blood glucose random) | Numeric | Linked to diabetic nephropathy |
| `bu` (blood urea) | Numeric | Nitrogenous waste retention |
| `sc` (serum creatinine) | Numeric | Glomerular filtration proxy |
| `htn` (hypertension) | Categorical | Major CKD risk factor |
| `dm` (diabetes mellitus) | Categorical | Leading cause of CKD |

### License

The dataset is publicly available for research use via the UCI repository.
Please cite:

> Dua, D. and Graff, C. (2019). UCI Machine Learning Repository.
> Irvine, CA: University of California, School of Information and Computer Science.

### How to Obtain the Data

1. Download from UCI: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease
2. Download directly from OpenML: `https://www.openml.org/d/40536`
3. Place the `chronic_kidney_disease.arff` file in the `data/` directory.

> The dataset is **not bundled** with this repository due to licence considerations.

---

## Results Summary

The following results are representative of what the pipeline produces on the
standard 400-sample UCI CKD dataset (70 / 15 / 15 split, random seed 42).
Your exact numbers may vary slightly depending on your scikit-learn version.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~96 % | ~97 % | ~97 % | ~97 % | ~0.99 |
| **Random Forest** | **~98 %** | **~98 %** | **~99 %** | **~98 %** | **~1.00** |
| SVM | ~97 % | ~97 % | ~99 % | ~98 % | ~0.99 |
| Gradient Boosting | ~97 % | ~97 % | ~98 % | ~98 % | ~1.00 |

**Best model:** Random Forest (or Gradient Boosting — both typically achieve near-perfect AUC on this dataset).

**Key findings:**
- Hemoglobin, specific gravity, and albumin are consistently the strongest predictors.
- All models significantly outperform a serum-creatinine-only threshold rule, validating the multi-biomarker approach.
- KNN imputation effectively handles the ~30 % missing-data rate without requiring row deletion.

---

## Project Structure

```
ckd_ml_project/
│
├── data/                          # Place your .arff dataset here
│   └── .gitkeep
│
├── models/                        # Saved model artefacts
│   └── ckd_pipeline.joblib        # Generated by run_pipeline.py
│
├── notebooks/                     # Jupyter notebooks (exploratory work)
│   └── .gitkeep
│
├── src/                           # Core Python source modules
│   ├── __init__.py
│   ├── config.py                  # Paths, seeds, hyper-param grids, logging
│   ├── data_loader.py             # ARFF loading → pandas DataFrame
│   ├── preprocess.py              # Cleaning, encoding, imputation, scaling, splitting
│   ├── feature_analysis.py        # EDA plots (correlation, importances, distribution)
│   ├── model_training.py          # GridSearchCV training for all four classifiers
│   ├── evaluation.py              # Metrics, ROC curves, confusion matrix plots
│   └── prediction.py             # CKDPredictor: serialisable inference wrapper
│
├── app/
│   ├── __init__.py
│   └── app.py                     # Streamlit 5-tab web application
│
├── logs/                          # Pipeline log file + saved PNG figures
│   └── pipeline.log               # Generated at runtime
│
├── run_pipeline.py                # Single-command end-to-end pipeline script
├── requirements.txt               # Pinned Python dependencies
└── README.md                      # This file
```

---

## Authors and Contributions

| Name | Role |
|---|---|
| *(Your Name)* | Data preprocessing pipeline, feature engineering, KNN imputation |
| *(Team Member 2)* | Model training module, hyperparameter tuning, GridSearchCV implementation |
| *(Team Member 3)* | Evaluation metrics, ROC/confusion matrix visualisations |
| *(Team Member 4)* | Streamlit application design, interactive CKD prediction interface |
| *(Team Member 5)* | Clinical context research, documentation, README, code review |

> Update the table above with your actual team members and their specific contributions before submission.

---

## Dependencies

All dependencies are pinned in `requirements.txt`.

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.3.3 | DataFrame manipulation |
| numpy | 2.4.2 | Numeric arrays |
| scipy | 1.17.1 | ARFF file parsing |
| scikit-learn | 1.8.0 | ML pipeline (imputation, scaling, models, metrics) |
| xgboost | 3.2.0 | Gradient Boosting classifier |
| matplotlib | 3.10.8 | Figure rendering |
| seaborn | 0.13.2 | Statistical visualisations |
| streamlit | 1.55.0 | Interactive web application |
| joblib | 1.5.3 | Model serialisation |

Install everything with:
```bash
pip install -r requirements.txt
```
