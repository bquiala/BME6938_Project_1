"""
model_training.py
=================
Train and hyper-parameter-tune four classifiers using GridSearchCV.

Models
------
- Logistic Regression  – interpretable baseline
- Random Forest        – non-linear ensemble, widely used in CKD research
- Support Vector Machine
- Gradient Boosting    – XGBoost when available, sklearn GBC as fallback

The public API is intentionally minimal:
    results = train_models(X_train, y_train, X_val, y_val)
    best_name, best_estimator = select_best_model(results)
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from src.config import CV_FOLDS, PARAM_GRIDS, RANDOM_STATE, get_logger

logger = get_logger(__name__)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_estimators() -> dict[str, Any]:
    """
    Instantiate all baseline estimators with the project's random seed.

    Returns
    -------
    dict mapping model name → unfitted estimator
    """
    estimators: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
        ),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": SVC(
            random_state=RANDOM_STATE,
            probability=True,  # needed for predict_proba / ROC-AUC
        ),
    }

    if _XGBOOST_AVAILABLE:
        estimators["GradientBoosting"] = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        )
        logger.info("Using XGBoost for GradientBoosting.")
    else:
        estimators["GradientBoosting"] = GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        )
        logger.info("XGBoost not available – using sklearn GradientBoostingClassifier.")

    return estimators


# ─── Public API ───────────────────────────────────────────────────────────────

def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Train all classifiers with ``GridSearchCV`` hyper-parameter tuning.

    Each model is fitted on *X_train / y_train*; validation accuracy is
    reported against *X_val / y_val* (not used for tuning – that uses
    ``CV_FOLDS``-fold cross-validation internally).

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training feature matrix and labels.
    X_val, y_val : np.ndarray
        Validation feature matrix and labels (held out from tuning).
    progress_callback : callable, optional
        Called after each model finishes training:
        ``progress_callback(model_name, step_index, total_models)``
        Useful for Streamlit progress bars.

    Returns
    -------
    dict
        Keys are model names; values are dicts with:

        ``best_estimator``  – fitted estimator with best hyper-parameters
        ``best_params``     – dict of best hyper-parameters
        ``cv_results``      – raw ``GridSearchCV.cv_results_`` dict
        ``val_score``       – accuracy on the validation split
        ``training_time``   – wall-clock seconds
    """
    estimators = _build_estimators()
    results: dict[str, dict[str, Any]] = {}
    total = len(estimators)

    for step, (name, estimator) in enumerate(estimators.items(), start=1):
        logger.info("[%d/%d] Training %s …", step, total, name)

        param_grid = PARAM_GRIDS.get(name, {})
        t0 = time.perf_counter()

        if param_grid:
            gs = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=CV_FOLDS,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X_train, y_train)
            best_estimator = gs.best_estimator_
            best_params    = gs.best_params_
            cv_results     = gs.cv_results_
        else:
            estimator.fit(X_train, y_train)
            best_estimator = estimator
            best_params    = {}
            cv_results     = {}

        elapsed   = time.perf_counter() - t0
        val_score = best_estimator.score(X_val, y_val)

        logger.info(
            "%s — best_params=%s  val_acc=%.4f  elapsed=%.1fs",
            name, best_params, val_score, elapsed,
        )

        results[name] = {
            "best_estimator": best_estimator,
            "best_params":    best_params,
            "cv_results":     cv_results,
            "val_score":      val_score,
            "training_time":  elapsed,
        }

        if progress_callback:
            progress_callback(name, step, total)

    return results


def select_best_model(
    results: dict[str, dict[str, Any]],
) -> tuple[str, Any]:
    """
    Select the model with the highest validation accuracy.

    Parameters
    ----------
    results : dict
        Output of :func:`train_models`.

    Returns
    -------
    (model_name, best_estimator)
        The name and the fitted estimator of the top-performing model.
    """
    best_name = max(results, key=lambda k: results[k]["val_score"])
    best_score = results[best_name]["val_score"]
    logger.info("Best model: %s (val_accuracy=%.4f)", best_name, best_score)
    return best_name, results[best_name]["best_estimator"]
