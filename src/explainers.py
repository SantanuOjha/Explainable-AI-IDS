"""SHAP and LIME explanation pipelines with latency benchmarking.

One hub object per loaded model. The hub is built once (cache it at the
app layer) so the expensive parts — the SHAP tree traversal structures
and the LIME training-data statistics — are paid a single time, not per
explanation request.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.utils import CLASS_NAMES

# Rows kept as the summarized background set for KernelSHAP / LIME.
# ponytail: fixed sizes tuned for interactive latency; make configurable if a
# non-tree model ever needs a bigger background.
_KERNEL_BACKGROUND_K = 25
_MAX_LIME_BACKGROUND = 2000


class ExplainerHub:
    """Wraps a fitted classifier with ready-to-use SHAP + LIME explainers.

    Parameters
    ----------
    model:
        Fitted binary classifier (RandomForest / XGBoost from the
        notebooks, or any estimator with predict_proba).
    background:
        Representative sample of the training/test distribution in the
        exact feature space the model was fit on. Used as the LIME
        training reference and, for non-tree models, the KernelSHAP
        background. A few hundred to a few thousand rows is plenty.
    """

    def __init__(self, model, background: pd.DataFrame):
        self.model = model
        self.feature_names = list(background.columns)
        # Cache a bounded background once; every explanation reuses it.
        self._background = background.sample(
            min(len(background), _MAX_LIME_BACKGROUND), random_state=0
        )

        try:
            # Exact, fast path for tree ensembles (RF, XGBoost).
            self._shap = shap.TreeExplainer(model)
            self.shap_algorithm = "TreeSHAP"
        except Exception:
            # Model-agnostic fallback: KernelSHAP over a kmeans-summarized
            # background to keep the sampling cost tractable.
            summary = shap.kmeans(self._background, _KERNEL_BACKGROUND_K)
            self._shap = shap.KernelExplainer(model.predict_proba, summary)
            self.shap_algorithm = "KernelSHAP"

        self._lime = LimeTabularExplainer(
            training_data=self._background.values,
            feature_names=self.feature_names,
            class_names=CLASS_NAMES,
            mode="classification",
            discretize_continuous=True,
            random_state=0,
        )

    # ------------------------------------------------------------------ SHAP

    def _attack_class_shap(self, values: np.ndarray) -> np.ndarray:
        """Reduce SHAP output to the ATTACK-class contribution matrix.

        sklearn ensembles yield (n, features, classes); XGBoost yields
        (n, features) in log-odds space. Both end up as (n, features).
        """
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        return values

    def _attack_base_value(self) -> float:
        base = np.ravel(np.asarray(self._shap.expected_value, dtype=np.float64))
        return float(base[1]) if base.size > 1 else float(base[0])

    def shap_values(self, x: pd.DataFrame) -> np.ndarray:
        """(n, features) SHAP contributions toward the ATTACK class."""
        return self._attack_class_shap(self._shap.shap_values(x.values))

    def global_importance(self, x: pd.DataFrame) -> pd.DataFrame:
        """Global feature importance: mean |SHAP value| over the sample x.

        Returns a DataFrame sorted descending with columns
        [feature, importance].
        """
        importance = np.abs(self.shap_values(x)).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def explain_shap(self, row: pd.Series) -> dict:
        """Local SHAP explanation for one flow, with wall-clock latency."""
        x = row.to_frame().T
        start = time.perf_counter()
        values = self.shap_values(x)[0]
        elapsed = time.perf_counter() - start
        return {
            "values": values,
            "base_value": self._attack_base_value(),
            "features": self.feature_names,
            "data": x.iloc[0].values,
            "algorithm": self.shap_algorithm,
            "latency_s": elapsed,
        }

    # ------------------------------------------------------------------ LIME

    def explain_lime(self, row: pd.Series, num_features: int = 10) -> dict:
        """Local LIME explanation for one flow, with wall-clock latency.

        Contributions are (condition, weight) pairs where weight > 0
        pushes toward ATTACK — same sign convention as the SHAP output.
        """
        start = time.perf_counter()
        exp = self._lime.explain_instance(
            row.values,
            self.model.predict_proba,
            num_features=num_features,
            labels=(1,),
        )
        elapsed = time.perf_counter() - start
        return {
            "contributions": exp.as_list(label=1),
            "algorithm": "LIME (local surrogate)",
            "latency_s": elapsed,
        }
