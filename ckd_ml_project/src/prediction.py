"""
prediction.py
=============
CKD risk prediction utilities.

The ``CKDPredictor`` class wraps a trained sklearn estimator together with
the preprocessing artefacts (scaler, imputer, encoders, LASSO feature mask)
into a single, serialisable object.  It accepts raw biomarker values from the
UI and returns a probability, a binary label, and a risk-level category.

The inference path mirrors the training pipeline exactly:
    raw inputs → impute (KNN) → Z-score scale → LASSO feature selection → predict

Typical usage
-------------
>>> predictor = CKDPredictor(model, scaler, all_feature_names, feature_mask,
...                          encoders, imputer)
>>> predictor.save()
>>> predictor = CKDPredictor.load()
>>> result    = predictor.predict({"hemoglobin": 11.0, "albumin": 2.0, ...})
>>> print(result["probability"], result["label"], result["risk_level"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, get_logger

logger = get_logger(__name__)

# Default artefact location
_DEFAULT_ARTIFACT = MODELS_DIR / "ckd_pipeline.joblib"

# Risk thresholds (CKD probability)
_THRESHOLD_HIGH     = 0.70
_THRESHOLD_MODERATE = 0.40


class CKDPredictor:
    """
    Serialisable wrapper for a trained CKD prediction pipeline.

    Attributes
    ----------
    model : fitted sklearn estimator
    scaler : fitted ``StandardScaler``
    all_feature_names : list[str]
        All feature names present after imputation (before LASSO selection).
    feature_mask : np.ndarray
        Boolean mask produced by LASSO selection; True = feature used by model.
    feature_names : list[str]
        LASSO-selected feature names (aligned with model input).
    encoders : dict[str, LabelEncoder]
        Fitted label encoders for categorical columns.
    imputer : fitted ``KNNImputer``
    """

    def __init__(
        self,
        model: Any,
        scaler: StandardScaler,
        all_feature_names: list[str],
        feature_mask: np.ndarray,
        encoders: dict,
        imputer: KNNImputer,
    ) -> None:
        self.model             = model
        self.scaler            = scaler
        self.all_feature_names = all_feature_names
        self.feature_mask      = feature_mask
        self.feature_names     = [
            fn for fn, keep in zip(all_feature_names, feature_mask) if keep
        ]
        self.encoders          = encoders
        self.imputer           = imputer

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Path = _DEFAULT_ARTIFACT) -> None:
        """
        Persist the predictor to disk using joblib.

        Parameters
        ----------
        path : Path
            Destination file.  Defaults to ``models/ckd_pipeline.joblib``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("CKDPredictor saved → %s", path)

    @classmethod
    def load(cls, path: Path = _DEFAULT_ARTIFACT) -> "CKDPredictor":
        """
        Load a serialised ``CKDPredictor`` from disk.

        Parameters
        ----------
        path : Path
            Source file.  Defaults to ``models/ckd_pipeline.joblib``.

        Returns
        -------
        CKDPredictor

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained predictor found at '{path}'.\n"
                "Run the pipeline first: python run_pipeline.py"
            )
        logger.info("Loading CKDPredictor from %s", path)
        return joblib.load(path)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Predict CKD risk for a single patient from raw biomarker values.

        The inference path mirrors training exactly:
        raw inputs → KNN impute → Z-score scale → LASSO feature mask → predict

        Missing features default to ``NaN`` and are imputed using the fitted
        KNNImputer — no special handling required in the UI.

        Parameters
        ----------
        inputs : dict
            Maps feature name (any of ``all_feature_names``) to a numeric or
            pre-encoded value.  Unknown keys are silently ignored.
            Missing features are treated as NaN and imputed.

        Returns
        -------
        dict
            ``probability`` – float in [0, 1], probability of CKD
            ``label``       – ``"CKD"`` or ``"No CKD"``
            ``risk_level``  – ``"High"`` | ``"Moderate"`` | ``"Low"``
        """
        # Build a single-row DataFrame aligned to ALL features (pre-LASSO)
        row = {feature: np.nan for feature in self.all_feature_names}
        row.update({k: v for k, v in inputs.items() if k in self.all_feature_names})

        X = pd.DataFrame([row])[self.all_feature_names]
        X = X.apply(pd.to_numeric, errors="coerce")

        # Step 1: KNN impute using the training imputer
        X_imputed = self.imputer.transform(X)

        # Step 2: Z-score scale with the training scaler
        X_scaled = self.scaler.transform(X_imputed)

        # Step 3: Apply the LASSO feature mask
        X_selected = X_scaled[:, self.feature_mask]

        # Step 4: Predict probability
        prob = float(self.model.predict_proba(X_selected)[0, 1])

        label = "CKD" if prob >= 0.5 else "No CKD"

        if prob >= _THRESHOLD_HIGH:
            risk_level = "High"
        elif prob >= _THRESHOLD_MODERATE:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        logger.debug(
            "Prediction — prob=%.4f  label=%s  risk=%s", prob, label, risk_level
        )
        return {"probability": prob, "label": label, "risk_level": risk_level}
