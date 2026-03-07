"""
app.py
======
Streamlit web application for the CKD Risk Prediction pipeline.

Launch
------
    streamlit run app/app.py

Tabs
----
🏠 Home            – project overview and clinical context
📂 Dataset         – upload / inspect the ARFF dataset
🔬 Training        – run the full ML pipeline with a progress indicator
📊 Visualisations  – correlation heatmap, ROC curves, confusion matrix,
                     feature importances, class distribution
🩺 Predict         – interactive single-patient CKD risk assessment
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, TARGET_COLUMN

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="CKD Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS – light / minimal aesthetic ───────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    .stApp { background-color: #F8F8F8; }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E4E4E4;
    }

    /* ── Metric card ── */
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        text-align: center;
        margin-bottom: 0.4rem;
    }
    .metric-card h3 {
        font-size: 0.78rem;
        color: #999;
        margin: 0 0 0.35rem;
        text-transform: uppercase;
        letter-spacing: .06em;
    }
    .metric-card p { font-size: 1.75rem; font-weight: 700; color: #333; margin: 0; }

    /* ── Risk badges ── */
    .risk-high     { color: #C0392B; font-weight: 700; font-size: 1.4rem; }
    .risk-moderate { color: #D35400; font-weight: 700; font-size: 1.4rem; }
    .risk-low      { color: #27AE60; font-weight: 700; font-size: 1.4rem; }

    /* ── Primary button ── */
    div.stButton > button {
        background: #E8A0B0;
        color: #333;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.8rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: background 0.2s;
    }
    div.stButton > button:hover { background: #D4849A; color: #fff; }

    /* ── Tabs ── */
    button[data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session-state initialisation ────────────────────────────────────────────
_STATE_DEFAULTS = {
    "df_raw":         None,
    "pipeline_data":  None,
    "train_results":  None,
    "best_model":     None,
    "best_model_name": None,
    "predictor":      None,
    "metrics_df":     None,
}
for _key, _default in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 CKD Risk Predictor")
    st.markdown("---")
    st.markdown(
        """
        **Chronic Kidney Disease** early-detection tool powered by machine learning.

        1. Upload a CKD dataset (`.arff`)
        2. Run the training pipeline
        3. Explore visualisations
        4. Predict individual patient risk
        """
    )
    st.markdown("---")
    st.caption("BME 6938 · Machine Learning in Medicine")

# ─── Tab layout ───────────────────────────────────────────────────────────────
tab_home, tab_data, tab_train, tab_viz, tab_predict = st.tabs(
    ["🏠 Home", "📂 Dataset", "🔬 Training", "📊 Visualisations", "🩺 Predict"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    col_l, col_r = st.columns([3, 1])

    with col_l:
        st.markdown(
            """
            ## Chronic Kidney Disease Risk Predictor

            A reproducible machine learning application for **early CKD detection** from
            routine clinical laboratory measurements and patient history.

            ---

            ### Clinical Context

            Chronic Kidney Disease (CKD) affects roughly **10 % of the global population**
            and is frequently detected late because a single biomarker such as serum
            creatinine may remain within normal range during early disease stages.

            This tool analyses **up to 24 clinical features simultaneously** — including
            haematological, biochemical, and demographic data — to estimate the probability
            that a patient has CKD.

            Target users: clinicians, healthcare researchers, and clinical data science students.

            ---

            ### Machine Learning Pipeline

            | Step | Module |
            |---|---|
            | ARFF data loading | `src/data_loader.py` |
            | Cleaning, imputation, encoding, scaling | `src/preprocess.py` |
            | Feature correlation & importance | `src/feature_analysis.py` |
            | Model training + GridSearchCV tuning | `src/model_training.py` |
            | Metrics & visualisations | `src/evaluation.py` |
            | Serialisable predictor | `src/prediction.py` |

            ---

            ### Models Compared

            | Model | Key Strength |
            |---|---|
            | Logistic Regression | Interpretable baseline |
            | Random Forest | Robust non-linear ensemble |
            | Support Vector Machine | High-dimensional generalisation |
            | Gradient Boosting (XGBoost) | State-of-the-art accuracy |
            """
        )

    with col_r:
        st.markdown("### Key Biomarkers")
        for icon, label in [
            ("🔴", "Hemoglobin"),
            ("💧", "Specific Gravity"),
            ("🟡", "Albumin"),
            ("🍬", "Blood Glucose"),
            ("🧪", "Blood Urea"),
            ("📈", "Serum Creatinine"),
            ("❤️", "Hypertension"),
            ("💉", "Diabetes Mellitus"),
            ("🩸", "Packed Cell Volume"),
            ("🔬", "White Blood Cells"),
        ]:
            st.markdown(f"{icon} &nbsp; {label}", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("Dataset Upload & Overview")
    st.markdown(
        "Upload the CKD dataset in **OpenML ARFF format** "
        "(`chronic_kidney_disease.arff`)."
    )

    col_up, col_sample = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload .arff file", type=["arff"], key="arff_uploader"
        )
    with col_sample:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        load_sample_btn = st.button("Load from data/ directory", key="btn_load_sample")

    if uploaded is not None:
        from src.data_loader import load_arff
        with st.spinner("Parsing ARFF …"):
            try:
                df = load_arff(uploaded)
                st.session_state["df_raw"] = df
                st.success(
                    f"Dataset loaded — **{df.shape[0]}** samples × **{df.shape[1]}** columns"
                )
            except Exception as exc:
                st.error(f"Failed to load dataset: {exc}")

    elif load_sample_btn:
        from src.data_loader import load_sample_data
        with st.spinner("Loading sample data …"):
            try:
                df = load_sample_data()
                st.session_state["df_raw"] = df
                st.success(
                    f"Sample dataset loaded — **{df.shape[0]}** samples × **{df.shape[1]}** columns"
                )
            except FileNotFoundError as exc:
                st.warning(str(exc))

    if st.session_state["df_raw"] is not None:
        df = st.session_state["df_raw"]

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples",   df.shape[0])
        c2.metric("Features",  df.shape[1] - 1)
        miss_pct = df.isnull().mean().mean() * 100
        c3.metric("Avg Missing %", f"{miss_pct:.1f}%")
        target_col = TARGET_COLUMN if TARGET_COLUMN in df.columns else df.columns[-1]
        ckd_count = (df[target_col].astype(str).str.strip().str.lower()
                     .isin({"ckd", "1", "yes"})).sum()
        c4.metric("CKD Positive", ckd_count)

        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        with st.expander("Missing Values per Feature"):
            miss = (df.isnull().mean() * 100).sort_values(ascending=False)
            miss = miss[miss > 0]
            if miss.empty:
                st.info("No missing values detected.")
            else:
                st.bar_chart(miss)

        with st.expander("Descriptive Statistics"):
            st.dataframe(df.describe(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.header("Model Training Pipeline")

    if st.session_state["df_raw"] is None:
        st.info("⬅  Please upload a dataset on the **Dataset** tab first.")
    else:
        st.markdown(
            "Click **Run Full Pipeline** to preprocess the data, train all models "
            "with hyper-parameter tuning, and evaluate them on the held-out test set."
        )

        if st.button("🚀 Run Full Pipeline", key="btn_run_pipeline"):
            progress_bar = st.progress(0, text="Initialising …")
            status_box   = st.empty()

            try:
                # ── Preprocess ────────────────────────────────────────────────
                status_box.markdown("⚙️ **Step 1 / 4 — Preprocessing data …**")
                progress_bar.progress(5, text="Preprocessing …")
                from src.preprocess import preprocess
                pipeline_data = preprocess(st.session_state["df_raw"].copy())
                st.session_state["pipeline_data"] = pipeline_data
                progress_bar.progress(15, text="Preprocessing complete.")

                # ── Train models ──────────────────────────────────────────────
                from src.model_training import train_models, select_best_model

                def _cb(name: str, step: int, total: int) -> None:
                    pct  = 15 + int(65 * step / total)
                    text = f"Training {name} ({step}/{total}) …"
                    progress_bar.progress(pct, text=text)
                    status_box.markdown(f"🔬 **Step 2 / 4 — {text}**")

                status_box.markdown("🔬 **Step 2 / 4 — Training models …**")
                train_results = train_models(
                    pipeline_data["X_train"], pipeline_data["y_train"],
                    pipeline_data["X_val"],   pipeline_data["y_val"],
                    progress_callback=_cb,
                )
                st.session_state["train_results"] = train_results

                # ── Select best ───────────────────────────────────────────────
                best_name, best_model = select_best_model(train_results)
                st.session_state["best_model_name"] = best_name
                st.session_state["best_model"]      = best_model

                # ── Evaluate ──────────────────────────────────────────────────
                status_box.markdown("📊 **Step 3 / 4 — Evaluating on test set …**")
                progress_bar.progress(85, text="Evaluating …")
                from src.evaluation import evaluate_all_models
                metrics_df = evaluate_all_models(
                    train_results,
                    pipeline_data["X_test"],
                    pipeline_data["y_test"],
                )
                st.session_state["metrics_df"] = metrics_df

                # ── Save predictor ────────────────────────────────────────────
                status_box.markdown("💾 **Step 4 / 4 — Saving model artefact …**")
                progress_bar.progress(95, text="Saving …")
                from src.prediction import CKDPredictor
                predictor = CKDPredictor(
                    model=best_model,
                    scaler=pipeline_data["scaler"],
                    feature_names=pipeline_data["feature_names"],
                    encoders=pipeline_data["encoders"],
                    imputer=pipeline_data["imputer"],
                )
                predictor.save()
                st.session_state["predictor"] = predictor

                progress_bar.progress(100, text="Complete!")
                status_box.success(
                    f"✅ Pipeline complete!  Best model: **{best_name}**"
                )

            except Exception as exc:
                progress_bar.empty()
                st.error(f"Pipeline error: {exc}")
                raise

        # ── Results ───────────────────────────────────────────────────────────
        if st.session_state["metrics_df"] is not None:
            st.markdown("---")
            st.subheader("Model Comparison — Test Set")

            metrics_df = st.session_state["metrics_df"]
            display_df = metrics_df.copy()
            for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map(lambda v: f"{v*100:.2f}%")
            st.dataframe(display_df, use_container_width=True)

            best = st.session_state["best_model_name"]
            st.success(f"🏆 **Best model selected: {best}**")

            st.subheader("Best Hyper-parameters per Model")
            for name, info in st.session_state["train_results"].items():
                with st.expander(name):
                    st.json(info["best_params"] if info["best_params"] else {"note": "No tuning grid defined."})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_viz:
    st.header("Visualisations")

    pipeline_data = st.session_state["pipeline_data"]
    train_results = st.session_state["train_results"]

    if pipeline_data is None:
        st.info("⬅  Run the training pipeline first to view visualisations.")
    else:
        from src.feature_analysis import (
            plot_class_distribution,
            plot_correlation_heatmap,
            plot_feature_importances,
        )
        from src.evaluation import plot_confusion_matrix, plot_roc_curve

        # ── Row 1: Class distribution + Correlation heatmap ───────────────────
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("Class Distribution")
            fig_cls = plot_class_distribution(pipeline_data["y_train"])
            st.pyplot(fig_cls, use_container_width=True)

        with col_b:
            st.subheader("Feature Correlation Heatmap")
            fig_corr = plot_correlation_heatmap(pipeline_data["df_clean"])
            st.pyplot(fig_corr, use_container_width=True)

        st.markdown("---")

        # ── Row 2: Feature importances ────────────────────────────────────────
        st.subheader("Top Feature Importances")
        fig_imp = plot_feature_importances(
            pipeline_data["feature_names"],
            pipeline_data["X_train"],
            pipeline_data["y_train"],
        )
        st.pyplot(fig_imp, use_container_width=True)

        st.markdown("---")

        # ── Row 3: ROC curves + Confusion matrix ──────────────────────────────
        if train_results is not None:
            col_c, col_d = st.columns(2)
            with col_c:
                st.subheader("ROC Curves — All Models")
                fig_roc = plot_roc_curve(
                    train_results,
                    pipeline_data["X_test"],
                    pipeline_data["y_test"],
                )
                st.pyplot(fig_roc, use_container_width=True)

            with col_d:
                best_name  = st.session_state["best_model_name"]
                best_model = st.session_state["best_model"]
                st.subheader(f"Confusion Matrix — {best_name}")
                fig_cm = plot_confusion_matrix(
                    best_model,
                    pipeline_data["X_test"],
                    pipeline_data["y_test"],
                    model_name=best_name,
                )
                st.pyplot(fig_cm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Interactive CKD Risk Prediction")

    # Try to load a previously saved predictor
    predictor = st.session_state.get("predictor")
    if predictor is None:
        artifact = MODELS_DIR / "ckd_pipeline.joblib"
        if artifact.exists():
            from src.prediction import CKDPredictor
            try:
                predictor = CKDPredictor.load(artifact)
                st.session_state["predictor"] = predictor
            except Exception:
                pass

    if predictor is None:
        st.info(
            "⬅  Train the pipeline first (Training tab) to enable predictions.\n\n"
            "Alternatively, run `python run_pipeline.py` from the terminal to train "
            "and save the model."
        )
    else:
        st.markdown(
            "Enter the patient's biomarker values. Leave fields at their defaults "
            "if a measurement is unavailable — missing values are imputed automatically."
        )

        # ── Numeric feature definitions (label, unit, min, max, default) ──────
        _NUMERIC_FEATURES: dict[str, tuple] = {
            "age":                    ("Age",                        "years",       0.0,    120.0,  50.0),
            "blood_pressure":         ("Blood Pressure",             "mm Hg",      50.0,   200.0,  80.0),
            "specific_gravity":       ("Specific Gravity",           "",          1.000,   1.030,  1.020),
            "albumin":                ("Albumin",                    "0–5 scale",   0.0,     5.0,   0.0),
            "sugar":                  ("Sugar",                      "0–5 scale",   0.0,     5.0,   0.0),
            "blood_glucose_random":   ("Blood Glucose Random",       "mg/dL",      50.0,   500.0, 120.0),
            "blood_urea":             ("Blood Urea",                 "mg/dL",       1.0,   400.0,  30.0),
            "serum_creatinine":       ("Serum Creatinine",           "mg/dL",       0.1,    80.0,   1.0),
            "sodium":                 ("Sodium",                     "mEq/L",     100.0,   170.0, 137.0),
            "potassium":              ("Potassium",                  "mEq/L",       2.0,    10.0,   4.5),
            "hemoglobin":             ("Hemoglobin",                 "g/dL",        2.0,    20.0,  14.0),
            "packed_cell_volume":     ("Packed Cell Volume",         "%",          10.0,    60.0,  44.0),
            "white_blood_cell_count": ("White Blood Cell Count",     "cells/cmm", 2000.0, 30000.0,8000.0),
            "red_blood_cell_count":   ("Red Blood Cell Count",       "M/cmm",       1.0,     7.0,   4.5),
        }

        # ── Categorical feature definitions (label, options) ──────────────────
        _CATEGORICAL_FEATURES: dict[str, tuple] = {
            "red_blood_cells":        ("Red Blood Cells",        ["normal", "abnormal"]),
            "pus_cell":               ("Pus Cell",               ["normal", "abnormal"]),
            "pus_cell_clumps":        ("Pus Cell Clumps",        ["notpresent", "present"]),
            "bacteria":               ("Bacteria",               ["notpresent", "present"]),
            "hypertension":           ("Hypertension",           ["no", "yes"]),
            "diabetes_mellitus":      ("Diabetes Mellitus",      ["no", "yes"]),
            "coronary_artery_disease":("Coronary Artery Disease",["no", "yes"]),
            "appetite":               ("Appetite",               ["good", "poor"]),
            "pedal_edema":            ("Pedal Edema",            ["no", "yes"]),
            "anemia":                 ("Anemia",                 ["no", "yes"]),
        }

        inputs: dict = {}

        with st.form("ckd_prediction_form"):
            # Numeric inputs
            st.subheader("Numeric Biomarkers")
            num_keys = [k for k in _NUMERIC_FEATURES if k in predictor.feature_names]
            for row_keys in [num_keys[i:i+3] for i in range(0, len(num_keys), 3)]:
                cols = st.columns(3)
                for col, key in zip(cols, row_keys):
                    label, unit, lo, hi, default = _NUMERIC_FEATURES[key]
                    display_label = f"{label}" + (f" ({unit})" if unit else "")
                    step = 0.001 if key == "specific_gravity" else 0.1
                    val = col.number_input(
                        display_label,
                        min_value=float(lo),
                        max_value=float(hi),
                        value=float(default),
                        step=step,
                        key=f"num_{key}",
                    )
                    inputs[key] = val

            # Categorical inputs
            st.subheader("Categorical Features")
            cat_keys = [k for k in _CATEGORICAL_FEATURES if k in predictor.feature_names]
            for row_keys in [cat_keys[i:i+3] for i in range(0, len(cat_keys), 3)]:
                cols = st.columns(3)
                for col, key in zip(cols, row_keys):
                    label, options = _CATEGORICAL_FEATURES[key]
                    choice = col.selectbox(label, options, key=f"cat_{key}")
                    # Encode using the fitted LabelEncoder if available
                    if key in predictor.encoders:
                        try:
                            enc_val = float(predictor.encoders[key].transform([choice])[0])
                        except ValueError:
                            enc_val = 0.0
                    else:
                        enc_val = float(options.index(choice))
                    inputs[key] = enc_val

            submitted = st.form_submit_button("🔍  Predict CKD Risk")

        # ── Display result ────────────────────────────────────────────────────
        if submitted:
            with st.spinner("Analysing biomarkers …"):
                result = predictor.predict(inputs)

            prob      = result["probability"]
            label     = result["label"]
            risk      = result["risk_level"]
            risk_cls  = f"risk-{risk.lower()}"

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"""<div class="metric-card">
                    <h3>CKD Probability</h3>
                    <p>{prob * 100:.1f}%</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                emoji = "🔴" if label == "CKD" else "🟢"
                st.markdown(
                    f"""<div class="metric-card">
                    <h3>Prediction</h3>
                    <p>{emoji} {label}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""<div class="metric-card">
                    <h3>Risk Level</h3>
                    <p class="{risk_cls}">{risk}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("#### Probability Gauge")
            st.progress(prob)

            if risk == "High":
                st.error(
                    "⚠️ **High CKD Risk** — Clinical evaluation and confirmatory testing "
                    "are strongly recommended."
                )
            elif risk == "Moderate":
                st.warning(
                    "⚠️ **Moderate CKD Risk** — Consider follow-up testing and lifestyle "
                    "modification counselling."
                )
            else:
                st.success(
                    "✅ **Low CKD Risk** — Biomarkers appear within a normal range. "
                    "Routine annual follow-up is advisable."
                )

            st.caption(
                "_This tool is intended for research and educational purposes only. "
                "It does not constitute medical advice._"
            )
