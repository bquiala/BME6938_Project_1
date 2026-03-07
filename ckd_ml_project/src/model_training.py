"""
model_training.py
=================
Train and hyper-parameter-tune four classifiers using GridSearchCV.

Models (per clinical specification)
------------------------------------
- Linear SVM (L2 penalty)  – robust baseline for structured clinical data
- Extra Trees              – fast, low-variance ensemble
- XGBoost                  – state-of-the-art gradient boosting
- LightGBM                 – fast gradient boosting on tabular data

CV scoring: Recall (Sensitivity) — minimises clinical false negatives.

The public API is intentionally minimal:
    results = train_models(X_train, y_train, X_val, y_val)
    best_name, best_estimator = select_best_model(results)
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

from src.config import CV_FOLDS, CV_SCORING, PARAM_GRIDS, RANDOM_STATE, get_logger

logger = get_logger(__name__)


# ─── Internal helpers ─────────────────────────────────────────────────────────

class _LinearSVCWrapper(BaseEstimator, ClassifierMixin):
    """
    Thin wrapper around ``LinearSVC`` that adds ``predict_proba`` via Platt scaling
    (``CalibratedClassifierCV``) so ROC-AUC and probability outputs work uniformly.
    Inherits ``BaseEstimator`` + ``ClassifierMixin`` for full scikit-learn compatibility.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        max_iter: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.random_state = random_state

    def _make_calibrated(self) -> Any:
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearSVC(
            C=self.C,
            penalty=self.penalty,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        return CalibratedClassifierCV(base, cv=5)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LinearSVCWrapper":
        self._clf = self._make_calibrated()
        self._clf.fit(X, y)
        self.classes_ = self._clf.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self._clf.score(X, y)


def _build_estimators() -> dict[str, Any]:
    """
    Instantiate all four clinical-specification estimators with the project's
    random seed.

    Returns
    -------
    dict mapping model name → unfitted estimator
    """
    estimators: dict[str, Any] = {
        # Linear SVM with L2 penalty — Platt-scaled for probability output
        "LinearSVM": _LinearSVCWrapper(
            C=1.0,
            penalty="l2",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        # Extra Trees — fast, low-variance ensemble
        "ExtraTrees": ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # XGBoost
    if _XGBOOST_AVAILABLE:
        estimators["XGBoost"] = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        )
        logger.info("XGBoost available — included in model suite.")
    else:
        logger.warning("XGBoost not installed — XGBoost model will be skipped.")

    # LightGBM
    if _LIGHTGBM_AVAILABLE:
        estimators["LightGBM"] = LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        logger.info("LightGBM available — included in model suite.")
    else:
        logger.warning("LightGBM not installed — LightGBM model will be skipped.")

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
    Train all four classifiers with ``GridSearchCV`` hyper-parameter tuning.

    Grid search uses ``CV_FOLDS``-fold (10-fold) cross-validation scored on
    **Recall** (sensitivity) as the primary metric, minimising clinical false
    negatives.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        SMOTE-balanced, LASSO-selected, Z-score-scaled training arrays.
    X_val, y_val : np.ndarray
        Validation feature matrix and labels (held out from grid search).
    progress_callback : callable, optional
        ``progress_callback(model_name, step_index, total_models)``

    Returns
    -------
    dict
        Keys are model names; values are dicts with:

        ``best_estimator``  – fitted estimator with best hyper-parameters
        ``best_params``     – dict of best hyper-parameters
        ``cv_results``      – raw ``GridSearchCV.cv_results_`` dict
        ``val_score``       – recall on the validation split
        ``val_accuracy``    – accuracy on the validation split
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
                scoring=CV_SCORING,   # recall — sensitivity
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

        elapsed      = time.perf_counter() - t0
        val_accuracy = best_estimator.score(X_val, y_val)

        # Compute recall on validation split
        from sklearn.metrics import recall_score
        y_val_pred = best_estimator.predict(X_val)
        val_recall = recall_score(y_val, y_val_pred, zero_division=0)

        logger.info(
            "%s — best_params=%s  val_recall=%.4f  val_acc=%.4f  elapsed=%.1fs",
            name, best_params, val_recall, val_accuracy, elapsed,
        )

        results[name] = {
            "best_estimator": best_estimator,
            "best_params":    best_params,
            "cv_results":     cv_results,
            "val_score":      val_recall,     # primary: recall
            "val_accuracy":   val_accuracy,
            "training_time":  elapsed,
        }

        if progress_callback:
            progress_callback(name, step, total)

    return results


def select_best_model(
    results: dict[str, dict[str, Any]],
) -> tuple[str, Any]:
    """
    Select the model with the highest validation **Recall** (sensitivity).

    Recall is the primary metric per the clinical specification: minimising
    false negatives is paramount in CKD screening.

    Parameters
    ----------
    results : dict
        Output of :func:`train_models`.

    Returns
    -------
    (model_name, best_estimator)
    """
    best_name  = max(results, key=lambda k: results[k]["val_score"])
    best_score = results[best_name]["val_score"]
    logger.info("Best model: %s (val_recall=%.4f)", best_name, best_score)
    return best_name, results[best_name]["best_estimator"]

