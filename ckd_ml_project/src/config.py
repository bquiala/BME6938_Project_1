"""
config.py
=========
Central configuration for the CKD Machine Learning pipeline.

All tunable constants – paths, random seeds, dataset parameters, model
hyper-parameter search spaces, and logging setup – live here so that the
rest of the codebase stays free of hard-coded values.
"""

from __future__ import annotations

import logging
from pathlib import Path

# ─── Directory layout ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # ckd_ml_project/
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure directories exist at import time
for _directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
    _directory.mkdir(parents=True, exist_ok=True)

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_STATE: int = 42

# ─── Dataset ─────────────────────────────────────────────────────────────────
TARGET_COLUMN: str = "class"
MISSING_THRESHOLD: float = 0.20   # drop features with > 20 % missing values
KNN_NEIGHBORS: int = 5            # neighbours used by KNNImputer

# ─── Data split ratios ───────────────────────────────────────────────────────
TRAIN_SIZE: float = 0.70          # 70 % for training
VALIDATION_SIZE: float = 0.50     # 50 % of the remaining 30 % → 15 % overall
TEST_SIZE: float = 0.50           # 50 % of the remaining 30 % → 15 % overall

# ─── Cross-validation ────────────────────────────────────────────────────────
CV_FOLDS: int = 5

# ─── Hyper-parameter search grids ────────────────────────────────────────────
PARAM_GRIDS: dict = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [1000],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
    },
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    },
}

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_FILE = LOGS_DIR / "pipeline.log"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with both console and rotating file handlers.

    Each logger is configured once; subsequent calls return the same instance
    without adding duplicate handlers.

    Parameters
    ----------
    name : str
        Dotted module name, e.g. ``src.preprocess``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
