# CKD Risk Predictor — Machine Learning Application

> A reproducible Python machine learning application that predicts **Chronic Kidney Disease (CKD) risk** from structured tabular clinical data using a multi-model pipeline with an interactive Streamlit interface.

---

## Table of Contents

1. [Clinical Context](#clinical-context)
2. [Running Locally — Step by Step](#running-locally--step-by-step)
3. [Troubleshooting](#troubleshooting)
4. [Usage Guide](#usage-guide)
5. [Data Description](#data-description)
6. [Results Summary](#results-summary)
7. [Project Structure](#project-structure)
8. [Authors and Contributions](#authors-and-contributions)
9. [Dependencies](#dependencies)

---

## Clinical Context

**Who is this for?**
Clinicians, healthcare researchers, and students studying clinical data science.

**What clinical problem does it address?**
Chronic Kidney Disease (CKD) affects roughly **10 % of the global population** and is often diagnosed late because individual biomarkers — most notably serum creatinine — can remain within normal reference ranges during early disease stages.

This application analyses **up to 24 clinical features simultaneously**, including haematological parameters (hemoglobin, packed cell volume, red/white blood cell counts), biochemical markers (albumin, blood urea, serum creatinine, blood glucose, sodium, potassium), urinalysis findings (specific gravity, albumin, sugar, red blood cells, pus cells, bacteria), and patient history (hypertension, diabetes mellitus, coronary artery disease, appetite, pedal edema, anemia).

By combining these features through a trained ensemble model, the system can detect CKD patterns that no single biomarker reveals in isolation.

---

## Development Disclaimer
This model was developed largely using the help from copilot's agent tool. GPT-5.4 was used to generate the plan/skeleton for the model, Claude Sonnet 4.6 was used to build out the majority of  the infrastructure, and Gemini 2.5 Pro was used for troubleshooting and pinpoint fixes. 

---

## Running Locally — Step by Step

### Step 1 — Install Git and Python

Download and install both tools before doing anything else.

- **Git**: https://git-scm.com/downloads
- **Python 3.11**: https://www.python.org/downloads/
  - Python 3.11 is recommended. 3.12+ may have minor compatibility friction with some pinned library versions.

> **Windows:** During the Python installer, tick **"Add Python to PATH"** or the `python` command will not be found in your terminal.

Verify the installs worked by opening a terminal and running:
```
git --version
python --version
```
Both should print a version number without errors.

---

### Step 2 — Clone the Repository

Open a terminal (see OS notes below) and run:

```bash
git clone https://github.com/bquiala/BME6938_Project_1.git
cd BME6938_Project_1/ckd_ml_project
```

> **Which terminal to use:**
> - **Windows** — use **Command Prompt** (`cmd.exe`). Search for it in the Start menu. Avoid PowerShell for this project (see Troubleshooting).
> - **macOS** — use **Terminal** (Applications → Utilities → Terminal).
> - **Linux** — any terminal emulator.

---

### Step 3 — Create a Virtual Environment

A virtual environment keeps this project's packages isolated from the rest of your machine. Run this once inside the `ckd_ml_project` folder:

```bash
python -m venv .venv
```

Then **activate** it. The command differs by OS:

| OS | Terminal | Activation command |
|---|---|---|
| Windows | Command Prompt (`cmd.exe`) | `.venv\Scripts\activate.bat` |
| Windows | PowerShell | See Troubleshooting — use cmd instead |
| macOS / Linux | Terminal | `source .venv/bin/activate` |

When the venv is active you will see `(.venv)` at the start of your prompt. You must activate it every time you open a new terminal window.

---

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

This downloads and installs all required packages (scikit-learn, XGBoost, LightGBM, Streamlit, etc.). It takes a few minutes the first time.

---

### Step 5 — Train the Model

The dataset is already included in `data/dataset.arff`. Run the full pipeline:

```bash
python run_pipeline.py
```

This preprocesses the data, trains four models with cross-validation, evaluates them on a held-out test set, and saves the best model to `models/ckd_pipeline.joblib`. You will see progress logs in the terminal. **Expected runtime: 3–10 minutes** depending on your hardware.

When complete you will see a results table:
```
=================================================================
  MODEL COMPARISON — Test Set
=================================================================
            accuracy  precision  recall     f1  roc_auc
Model
LinearSVM     1.0000     1.0000  1.0000 1.0000   1.0000
ExtraTrees    0.9833     0.9737  1.0000 0.9867   1.0000
...
=================================================================
```

---

### Step 6 — Launch the Web App

```bash
streamlit run app/app.py
```

Your browser will open automatically at **http://localhost:8501**. If it does not, open that URL manually.

---

### Every Time You Return to the Project

```bash
# Windows (cmd)
cd BME6938_Project_1\ckd_ml_project
.venv\Scripts\activate.bat
streamlit run app/app.py

# macOS / Linux
cd BME6938_Project_1/ckd_ml_project
source .venv/bin/activate
streamlit run app/app.py
```

You do not need to run `pip install` or `python run_pipeline.py` again unless you have deleted the venv or want to retrain the model.

---

**Computational requirements:** any modern CPU; no GPU required. 8 GB RAM recommended.

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

Data downloaded directly from OpenML: `https://www.openml.org/d/40536`

## Results Summary

The following results are representative of what the pipeline produces on the
standard 400-sample UCI CKD dataset (70 / 15 / 15 split, random seed 42).
Your exact numbers may vary slightly depending on your scikit-learn version.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~98.3 % | ~97.4 % | ~100 % | ~98.7 % | ~1.00 |
| Random Forest | ~98.3 % | ~97.4 % | ~100 % | ~98.7 % | ~1.00 |
| SVM | ~100 % | ~100 % | ~100 % | ~100 % | ~1.00 |
| Gradient Boosting | ~98.3 % | ~97.4 % | ~100 % | ~98.7 % | ~0.99 |

**Highest performing model:** Linear SVM

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
| *(Quinn Mullings)* | Model delvelopment and implementation, analysis of methods and data, results and evidence
| *(Bryan Quiala)* | Literature Review, Discussion & Limitations
| *(James Garner)* | Abstract, Introduction, Results and Evidence

---

## Troubleshooting

### Step 1 — Git / Python not found after installing

| OS | Fix |
|---|---|
| Windows | Re-run the Python installer, tick **"Add Python to PATH"**, then restart your terminal |
| macOS | Install via `brew install python@3.11` if the system Python is too old |

---

### Step 3 — Virtual environment activation fails on Windows PowerShell

**Error:**
```
File .venv\Scripts\Activate.ps1 cannot be loaded because running scripts
is disabled on this system.
```

**Cause:** PowerShell blocks unsigned scripts by default.

**Fix A (recommended) — switch to Command Prompt:**
Open `cmd.exe` instead of PowerShell and run:
```cmd
.venv\Scripts\activate.bat
```

**Fix B — change the PowerShell execution policy:**
Open PowerShell **as Administrator** and run once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then re-run `.venv\Scripts\activate` in a normal PowerShell window.

---

### Step 3 — `source` is not a recognised command (Windows)

**Error:**
```
'source' is not recognized as an internal or external command
```

**Cause:** `source` is a macOS/Linux shell built-in. It does not exist on Windows.

**Fix:** Use the Windows-specific command instead:
```cmd
.venv\Scripts\activate.bat
```

---

### Step 4 — LightGBM install fails on Windows

**Cause:** LightGBM requires the Microsoft Visual C++ Redistributable.

**Fix:** Download and install it from https://aka.ms/vs/17/release/vc_redist.x64.exe, then re-run `pip install -r requirements.txt`.

---

### Step 5 — `python` opens the Windows Store instead of running

**Fix:** During the Python installer, make sure **"Add Python to PATH"** was checked. If not, uninstall and reinstall Python with that option ticked. Alternatively, try using `py` instead of `python`:
```cmd
py run_pipeline.py
```

---

### Step 6 — `streamlit: command not found`

**Cause:** The virtual environment is not activated, so the `streamlit` executable is not on your PATH.

**Fix:** Activate the venv first (Step 3), then run the streamlit command.

---

### Step 6 — Port 8501 already in use

If another Streamlit instance is already running, start the app on a different port:
```bash
streamlit run app/app.py --server.port 8502
```
Then navigate to **http://localhost:8502**.

---

### Step 6 — "No trained predictor found" error in the app

The app requires a trained model file at `models/ckd_pipeline.joblib`. Run the pipeline first:
```bash
python run_pipeline.py
```

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
