"""
feature_analysis.py
===================
Exploratory data analysis and feature visualisation utilities.

Functions
---------
plot_correlation_heatmap    – Pearson correlation heatmap of all numeric features
plot_feature_importances    – Random Forest-derived feature importance bar chart
plot_class_distribution     – Bar chart of CKD vs. non-CKD sample counts
plot_lasso_coefficients     – LASSO coefficient magnitude bar chart
plot_pca_scatter            – 2-D PCA scatter plot coloured by class label
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for both CLI and Streamlit)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from src.config import HIGH_PRIORITY_FEATURES, RANDOM_STATE, TARGET_COLUMN, get_logger

logger = get_logger(__name__)

# ─── Shared colour palette ────────────────────────────────────────────────────
_PALETTE = {
    "background": "#FFFFFF",
    "grid":       "#F0F0F0",
    "pink":       "#E8A0B0",   # light-pink accent
    "silver":     "#B0B8C8",   # silver-blue
    "text":       "#333333",
}

sns.set_theme(style="whitegrid", font_scale=1.0)


# ─── Plotting functions ──────────────────────────────────────────────────────

def plot_correlation_heatmap(df_clean: pd.DataFrame) -> plt.Figure:
    """
    Generate a lower-triangle Pearson correlation heatmap.

    The figure uses the project's light-pink colour map so it integrates
    seamlessly with the Streamlit and report aesthetics.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned, imputed DataFrame (pre-scaling) that contains the target
        column.  The target is included in the correlation so clinicians can
        see which features correlate most strongly with CKD.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Generating correlation heatmap …")
    corr = df_clean.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(13, 10), facecolor=_PALETTE["background"])
    ax.set_facecolor(_PALETTE["background"])

    # Mask the upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdPu",
        linewidths=0.4,
        ax=ax,
        cbar_kws={"shrink": 0.75},
        annot_kws={"size": 7},
    )
    ax.set_title(
        "Feature Correlation Heatmap",
        fontsize=14, color=_PALETTE["text"], pad=15,
    )
    ax.tick_params(colors=_PALETTE["text"])
    plt.tight_layout()
    return fig


def plot_feature_importances(
    feature_names: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    top_n: int = 15,
) -> plt.Figure:
    """
    Fit a shallow Random Forest and plot the top-N feature importances.

    A dedicated shallow Random Forest (separate from the tuned pipeline model)
    is fitted here purely to derive stable Gini importances for visualisation.

    Parameters
    ----------
    feature_names : list[str]
        Names aligned with the columns of *X_train*.
    X_train : np.ndarray
        Scaled training feature matrix.
    y_train : np.ndarray
        Training labels.
    top_n : int
        Number of top features to display.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Computing feature importances (top %d) …", top_n)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = (
        pd.Series(rf.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
    )
    top = importances.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=_PALETTE["background"])
    ax.set_facecolor(_PALETTE["background"])

    colors = sns.color_palette("RdPu", top_n)
    # Reverse so the most important feature is at the top of a horizontal bar
    top[::-1].plot(kind="barh", ax=ax, color=colors[::-1])

    ax.set_title(
        f"Top {top_n} Feature Importances (Random Forest)",
        fontsize=13, color=_PALETTE["text"],
    )
    ax.set_xlabel("Gini Importance Score", color=_PALETTE["text"])
    ax.tick_params(colors=_PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["grid"])
    plt.tight_layout()
    return fig


def plot_class_distribution(y: np.ndarray) -> plt.Figure:
    """
    Plot the class-label distribution as a bar chart.

    Provides a quick visual check of class imbalance before modelling.

    Parameters
    ----------
    y : np.ndarray
        Integer label array (0 = No CKD, 1 = CKD).

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Plotting class distribution …")
    counts = pd.Series(y).value_counts().sort_index()
    labels = {0: "No CKD", 1: "CKD"}
    x_labels = [labels.get(int(k), str(k)) for k in counts.index]

    fig, ax = plt.subplots(figsize=(5, 4), facecolor=_PALETTE["background"])
    ax.set_facecolor(_PALETTE["background"])

    bar_colors = [_PALETTE["silver"], _PALETTE["pink"]][:len(counts)]
    bars = ax.bar(x_labels, counts.values, color=bar_colors, width=0.5, edgecolor="white")

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            str(val),
            ha="center", va="bottom",
            color=_PALETTE["text"], fontsize=11, fontweight="bold",
        )

    ax.set_title("Class Distribution", fontsize=13, color=_PALETTE["text"])
    ax.set_ylabel("Sample Count", color=_PALETTE["text"])
    ax.tick_params(colors=_PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["grid"])
    plt.tight_layout()
    return fig


def plot_lasso_coefficients(
    all_feature_names: list[str],
    lasso_coef: np.ndarray,
    feature_mask: np.ndarray,
    top_n: int = 20,
) -> plt.Figure:
    """
    Plot LASSO (L1) logistic regression coefficient magnitudes.

    Features selected by LASSO (coef > 0) are highlighted.  High-priority
    clinical features defined in ``config.HIGH_PRIORITY_FEATURES`` are
    additionally annotated with a star (★).

    Parameters
    ----------
    all_feature_names : list[str]
        Feature names aligned with *lasso_coef*.
    lasso_coef : np.ndarray
        Absolute coefficient values from the fitted LASSO estimator.
    feature_mask : np.ndarray
        Boolean array: True = feature was selected by LASSO.
    top_n : int
        Maximum number of features to display.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Plotting LASSO coefficient magnitudes …")
    series = (
        pd.Series(lasso_coef, index=all_feature_names)
        .sort_values(ascending=False)
        .head(top_n)
    )
    names = series.index.tolist()
    values = series.values

    # Colour: pink if selected, silver if zeroed out by LASSO
    selected_set = {n for n, m in zip(all_feature_names, feature_mask) if m}
    bar_colors = [
        _PALETTE["pink"] if n in selected_set else _PALETTE["silver"]
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=_PALETTE["background"])
    ax.set_facecolor(_PALETTE["background"])

    # Horizontal bars (reversed so highest is at top)
    y_pos = np.arange(len(names))[::-1]
    ax.barh(y_pos, values, color=bar_colors, edgecolor="white")

    # Annotate high-priority clinical features with ★
    high_priority = set(HIGH_PRIORITY_FEATURES)
    tick_labels = []
    for name in names[::-1]:           # bottom → top order
        label = name.replace("_", " ").title()
        if name in high_priority:
            label = f"★ {label}"
        tick_labels.append(label)

    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(tick_labels, fontsize=9, color=_PALETTE["text"])
    ax.set_xlabel("|LASSO Coefficient|", color=_PALETTE["text"])
    ax.set_title(
        "LASSO Feature Selection — Coefficient Magnitudes\n"
        "(Pink = selected  ·  ★ = high-priority clinical feature)",
        fontsize=12, color=_PALETTE["text"],
    )
    ax.tick_params(colors=_PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["grid"])
    plt.tight_layout()
    return fig


def plot_pca_scatter(
    X_scaled: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
) -> plt.Figure:
    """
    Project the (full-feature) scaled matrix onto the first two PCA axes and
    draw a scatter plot coloured by class label.

    This gives clinicians an intuitive view of separability in the reduced
    feature space.

    Parameters
    ----------
    X_scaled : np.ndarray
        Full-feature, Z-score-scaled training matrix (post-SMOTE).
    y : np.ndarray
        Class labels (0 = No CKD, 1 = CKD).
    n_components : int
        Number of PCA components (default 2 for 2-D scatter).

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Computing PCA (%d components) …", n_components)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=_PALETTE["background"])
    ax.set_facecolor(_PALETTE["background"])

    colours = {0: _PALETTE["silver"], 1: _PALETTE["pink"]}
    labels  = {0: "No CKD", 1: "CKD"}

    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colours[cls], label=labels[cls],
            alpha=0.65, edgecolors="white", linewidths=0.3, s=40,
        )

    ax.set_xlabel(f"PC 1  ({var_explained[0]:.1f}% variance)", color=_PALETTE["text"])
    ax.set_ylabel(f"PC 2  ({var_explained[1]:.1f}% variance)", color=_PALETTE["text"])
    ax.set_title(
        f"PCA — 2D Projection (total variance explained: "
        f"{sum(var_explained[:2]):.1f}%)",
        fontsize=13, color=_PALETTE["text"],
    )
    ax.legend(fontsize=10)
    ax.tick_params(colors=_PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["grid"])
    plt.tight_layout()
    return fig
