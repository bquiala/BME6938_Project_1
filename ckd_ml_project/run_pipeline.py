"""
run_pipeline.py
===============
End-to-end CKD Machine Learning pipeline — single-command entry point.

Usage
-----
    python run_pipeline.py                         # uses first .arff in data/
    python run_pipeline.py --data path/to/file.arff

Steps
-----
1.  Load the CKD ARFF dataset.
2.  Preprocess: clean, encode, impute, scale, split (70 / 15 / 15).
3.  Generate exploratory figures (class distribution, correlation, importances).
4.  Train four models with GridSearchCV hyper-parameter tuning.
5.  Evaluate all models on the held-out test set.
6.  Print a summary table to stdout.
7.  Generate evaluation figures (ROC curves, confusion matrix).
8.  Save the best model + preprocessing artefacts as ``models/ckd_pipeline.joblib``.

All figures are saved as PNG files in ``logs/``.
All pipeline activity is logged to ``logs/pipeline.log`` and to the console.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

# ── Ensure the project root is importable ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LOGS_DIR, MODELS_DIR, get_logger
from src.data_loader import load_arff, load_sample_data
from src.evaluation import evaluate_all_models, plot_confusion_matrix, plot_roc_curve
from src.feature_analysis import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_importances,
)
from src.model_training import select_best_model, train_models
from src.prediction import CKDPredictor
from src.preprocess import preprocess

logger = get_logger("run_pipeline")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save_figure(fig: plt.Figure, name: str) -> None:
    """Save a matplotlib Figure as a high-resolution PNG in ``logs/``."""
    path = LOGS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", path)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main(data_path: str | None = None) -> None:
    """
    Execute the complete CKD ML pipeline end-to-end.

    Parameters
    ----------
    data_path : str | None
        Path to a ``.arff`` dataset file.  If ``None``, the first ``.arff``
        file found in ``data/`` is used.
    """
    logger.info("=" * 65)
    logger.info("  CKD Machine Learning Pipeline  —  Start")
    logger.info("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Step 1/7  Loading dataset …")
    if data_path:
        df = load_arff(Path(data_path))
    else:
        df = load_sample_data()
    logger.info("Raw dataset shape: %s", df.shape)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    logger.info("Step 2/7  Preprocessing …")
    pipeline_data = preprocess(df.copy())

    logger.info(
        "Train: %d  |  Val: %d  |  Test: %d",
        len(pipeline_data["y_train"]),
        len(pipeline_data["y_val"]),
        len(pipeline_data["y_test"]),
    )

    # ── 3. Exploratory figures ────────────────────────────────────────────────
    logger.info("Step 3/7  Generating exploratory figures …")
    _save_figure(
        plot_class_distribution(pipeline_data["y_train"]),
        "class_distribution",
    )
    _save_figure(
        plot_correlation_heatmap(pipeline_data["df_clean"]),
        "correlation_heatmap",
    )
    _save_figure(
        plot_feature_importances(
            pipeline_data["feature_names"],
            pipeline_data["X_train"],
            pipeline_data["y_train"],
        ),
        "feature_importances",
    )

    # ── 4. Train models ───────────────────────────────────────────────────────
    logger.info("Step 4/7  Training models with GridSearchCV …")
    train_results = train_models(
        pipeline_data["X_train"], pipeline_data["y_train"],
        pipeline_data["X_val"],   pipeline_data["y_val"],
    )

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    logger.info("Step 5/7  Evaluating on held-out test set …")
    metrics_df = evaluate_all_models(
        train_results,
        pipeline_data["X_test"],
        pipeline_data["y_test"],
    )

    # Print results table
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON — Test Set")
    print("=" * 65)
    print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 65 + "\n")

    # ── 6. Evaluation figures ─────────────────────────────────────────────────
    logger.info("Step 6/7  Generating evaluation figures …")
    _save_figure(
        plot_roc_curve(
            train_results,
            pipeline_data["X_test"],
            pipeline_data["y_test"],
        ),
        "roc_curves",
    )

    best_name, best_model = select_best_model(train_results)
    _save_figure(
        plot_confusion_matrix(
            best_model,
            pipeline_data["X_test"],
            pipeline_data["y_test"],
            model_name=best_name,
        ),
        "confusion_matrix_best",
    )

    # ── 7. Save predictor artefact ────────────────────────────────────────────
    logger.info("Step 7/7  Saving model artefact …")
    predictor = CKDPredictor(
        model=best_model,
        scaler=pipeline_data["scaler"],
        feature_names=pipeline_data["feature_names"],
        encoders=pipeline_data["encoders"],
        imputer=pipeline_data["imputer"],
    )
    predictor.save()

    logger.info("=" * 65)
    logger.info("  Pipeline complete.  Best model: %s", best_name)
    logger.info(
        "  Best model metrics (test): acc=%.4f  f1=%.4f  auc=%.4f",
        metrics_df.loc[best_name, "accuracy"],
        metrics_df.loc[best_name, "f1"],
        metrics_df.loc[best_name, "roc_auc"],
    )
    logger.info("  Figures saved to : %s", LOGS_DIR)
    logger.info("  Model artefact   : %s/ckd_pipeline.joblib", MODELS_DIR)
    logger.info("=" * 65)
    logger.info("  Launch the Streamlit app with:")
    logger.info("      streamlit run app/app.py")
    logger.info("=" * 65)


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CKD Machine Learning Pipeline — end-to-end training and evaluation."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the .arff dataset file. "
            "Defaults to the first .arff file found in data/."
        ),
    )
    args = parser.parse_args()
    main(data_path=args.data)
