"""
evaluation.py
=============
Model evaluation metrics and publication-quality visualisations.

Functions
---------
compute_metrics       – accuracy, precision, recall, F1, ROC-AUC for one model
evaluate_all_models   – tabular comparison across all trained models
plot_roc_curve        – overlaid ROC curves for all models
plot_confusion_matrix – annotated confusion-matrix heatmap for the best model
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import get_logger

logger = get_logger(__name__)

# ─── Shared aesthetics ────────────────────────────────────────────────────────
_PALETTE = {
    "background": "#FFFFFF",
    "grid":       "#F0F0F0",
    "pink":       "#E8A0B0",
    "blue":       "#6B9AC4",
    "text":       "#333333",
}


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    split_name: str = "Test",
) -> dict[str, float]:
    """
    Compute binary classification metrics for a fitted estimator.

    Parameters
    ----------
    model : fitted sklearn estimator
        Must implement ``predict``.  ``predict_proba`` is used for ROC-AUC
        when available; falls back to ``decision_function`` otherwise.
    X : np.ndarray
        Feature matrix.
    y_true : np.ndarray
        Ground-truth binary labels.
    split_name : str
        Label used in log output (e.g. ``"Train"``, ``"Val"``, ``"Test"``).

    Returns
    -------
    dict
        ``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``
    """
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
    else:
        y_score = None

    metrics: dict[str, float] = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_score) if y_score is not None else float("nan"),
    }

    logger.info(
        "[%s] acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
        split_name,
        metrics["accuracy"], metrics["precision"],
        metrics["recall"],   metrics["f1"],   metrics["roc_auc"],
    )
    return metrics


def evaluate_all_models(
    results: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate every trained model on the held-out test set.

    Parameters
    ----------
    results : dict
        Output of :func:`src.model_training.train_models`.
    X_test, y_test : np.ndarray
        Held-out test data (never seen during training or hyper-parameter search).

    Returns
    -------
    pd.DataFrame
        One row per model, columns: accuracy, precision, recall, f1, roc_auc,
        Training Time (s).  Index is the model name.
    """
    rows = []
    for name, info in results.items():
        m = compute_metrics(info["best_estimator"], X_test, y_test, split_name=name)
        rows.append({
            "Model":              name,
            "accuracy":           m["accuracy"],
            "precision":          m["precision"],
            "recall":             m["recall"],
            "f1":                 m["f1"],
            "roc_auc":            m["roc_auc"],
            "Training Time (s)":  round(info["training_time"], 2),
        })
    return pd.DataFrame(rows).set_index("Model")


# ─── Visualisations ───────────────────────────────────────────────────────────

def plot_roc_curve(
    results: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> plt.Figure:
    """
    Plot overlaid ROC curves for all trained models.

    Parameters
    ----------
    results : dict
        Output of :func:`src.model_training.train_models`.
    X_test, y_test : np.ndarray
        Held-out test data.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Generating ROC curves …")
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=_PALETTE["background"])
    ax.set_facecolor(_PALETTE["background"])

    colors = sns.color_palette("Set2", len(results))
    for (name, info), color in zip(results.items(), colors):
        model = info["best_estimator"]
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            logger.warning("%s does not support probability output – skipping ROC.", name)
            continue

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {roc_auc:.3f})", linewidth=2, color=color)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", color=_PALETTE["text"])
    ax.set_ylabel("True Positive Rate",  color=_PALETTE["text"])
    ax.set_title("ROC Curves — All Models", fontsize=13, color=_PALETTE["text"])
    ax.legend(loc="lower right", fontsize=9)
    ax.tick_params(colors=_PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["grid"])
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Best Model",
) -> plt.Figure:
    """
    Plot an annotated confusion-matrix heatmap.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test, y_test : np.ndarray
        Held-out test data.
    model_name : str
        Shown in the figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Generating confusion matrix for %s …", model_name)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=_PALETTE["background"])
    ax.set_facecolor(_PALETTE["background"])

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdPu",
        xticklabels=["No CKD", "CKD"],
        yticklabels=["No CKD", "CKD"],
        ax=ax,
        linewidths=0.5,
        cbar=False,
    )
    ax.set_xlabel("Predicted Label", color=_PALETTE["text"])
    ax.set_ylabel("True Label",      color=_PALETTE["text"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, color=_PALETTE["text"])
    ax.tick_params(colors=_PALETTE["text"])
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC
    across all trained models on the held-out test set.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of :func:`evaluate_all_models` (index = model name).

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Generating metrics comparison chart …")

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    present_cols = [c for c in metric_cols if c in metrics_df.columns]
    display_names = {
        "accuracy":  "Accuracy",
        "precision": "Precision",
        "recall":    "Recall",
        "f1":        "F1-Score",
        "roc_auc":   "ROC-AUC",
    }

    models = metrics_df.index.tolist()
    x = np.arange(len(present_cols))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=_PALETTE["background"])
    ax.set_facecolor(_PALETTE["background"])

    colors = sns.color_palette("Set2", len(models))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [float(metrics_df.loc[model, c]) for c in present_cols]
        offsets = x + (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(offsets, vals, width * 0.92, label=model, color=color)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.007,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=_PALETTE["text"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_names[c] for c in present_cols],
        color=_PALETTE["text"], fontsize=11,
    )
    ax.set_ylabel("Score", color=_PALETTE["text"])
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Classification Metrics — All Models (Test Set)",
        fontsize=13, color=_PALETTE["text"],
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(colors=_PALETTE["text"])
    ax.yaxis.grid(True, color=_PALETTE["grid"], linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["grid"])
    plt.tight_layout()
    return fig


def plot_cv_scores(train_results: dict) -> plt.Figure:
    """
    Box plot of per-fold cross-validation recall scores for each model,
    drawn from the best hyper-parameter combination found during GridSearchCV.

    Parameters
    ----------
    train_results : dict
        Output of :func:`src.model_training.train_models`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Generating cross-validation scores chart …")

    model_names: list[str] = []
    fold_scores_list: list[list[float]] = []

    for name, info in train_results.items():
        cv_res = info.get("cv_results", {})

        if not cv_res:
            # No grid search was run — use val_score as a single data point
            model_names.append(name)
            fold_scores_list.append([float(info["val_score"])])
            continue

        mean_scores = cv_res.get("mean_test_score", np.array([]))
        if len(mean_scores) == 0:
            model_names.append(name)
            fold_scores_list.append([float(info["val_score"])])
            continue

        best_idx = int(np.argmax(mean_scores))
        fold_keys = sorted(
            k for k in cv_res
            if k.startswith("split") and k.endswith("_test_score")
        )
        if fold_keys:
            fold_scores = [float(cv_res[k][best_idx]) for k in fold_keys]
        else:
            fold_scores = [float(mean_scores[best_idx])]

        model_names.append(name)
        fold_scores_list.append(fold_scores)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_PALETTE["background"])
    ax.set_facecolor(_PALETTE["background"])

    colors = sns.color_palette("Set2", len(model_names))
    bp = ax.boxplot(
        fold_scores_list,
        patch_artist=True,
        notch=False,
        medianprops=dict(color=_PALETTE["text"], linewidth=2),
        whiskerprops=dict(color=_PALETTE["text"]),
        capprops=dict(color=_PALETTE["text"]),
        flierprops=dict(marker="o", markerfacecolor=_PALETTE["pink"], markersize=5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    rng = np.random.default_rng(42)
    for i, (scores, color) in enumerate(zip(fold_scores_list, colors), start=1):
        jitter = rng.uniform(-0.12, 0.12, len(scores))
        ax.scatter(
            np.full(len(scores), i) + jitter,
            scores,
            color=color, s=22, zorder=3, alpha=0.85, edgecolors="white", linewidths=0.4,
        )

    ax.set_xticks(range(1, len(model_names) + 1))
    ax.set_xticklabels(model_names, color=_PALETTE["text"], fontsize=10)
    ax.set_ylabel("Recall (CV Fold)", color=_PALETTE["text"])
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Cross-Validation Recall Scores — Best Hyper-parameter Set per Model",
        fontsize=13, color=_PALETTE["text"],
    )
    ax.tick_params(colors=_PALETTE["text"])
    ax.yaxis.grid(True, color=_PALETTE["grid"], linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["grid"])
    plt.tight_layout()
    return fig
