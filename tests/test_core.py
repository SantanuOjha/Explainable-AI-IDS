"""Unit tests for the data layer, evaluation and explainer pipelines.

The Data/ directory is gitignored, so everything here runs on a small
synthetic dataset shaped exactly like the CICIDS2017 feature space
(78 features, binary label). Tests that need the real artifacts are
skipped automatically when Data/ is absent (e.g. in CI).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import evaluate_model, predict_one
from src.explainers import ExplainerHub
from src.utils import DATA_DIR, FEATURE_NAMES, load_test_data, sanitize_features

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def synthetic():
    """Tiny CICIDS2017-shaped dataset + fitted RF, shared across tests."""
    x = pd.DataFrame(RNG.normal(size=(300, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    # Make the label learnable from the first feature so metrics are non-trivial.
    y = pd.Series((x.iloc[:, 0] > 0).astype(int))
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x.values, y)
    return x, y, model


def test_sanitize_features_handles_categorical_and_inf():
    dirty = pd.DataFrame({"a": ["tcp", "1.5", None], "b": [np.inf, -np.inf, 2.0]})
    clean = sanitize_features(dirty)
    assert clean.to_numpy().tolist() == [[0.0, 0.0], [1.5, 0.0], [0.0, 2.0]]
    assert np.isfinite(clean.to_numpy()).all()


def test_evaluate_model_returns_all_metrics(synthetic):
    x, y, model = synthetic
    m = evaluate_model(model, x, y)
    assert set(m) >= {"accuracy", "precision", "recall", "f1", "latency_per_sample_us"}
    assert m["accuracy"] > 0.9  # trivially learnable target
    assert m["confusion_matrix"].shape == (2, 2)
    assert m["latency_batch_s"] > 0


def test_predict_one_probabilities_sum_to_one(synthetic):
    x, _, model = synthetic
    pred = predict_one(model, x.iloc[0])
    assert pred["label"] in (0, 1)
    assert pred["proba_benign"] + pred["proba_attack"] == pytest.approx(1.0)
    assert pred["latency_ms"] > 0


def test_explainer_hub_shap_and_lime(synthetic):
    x, _, model = synthetic
    hub = ExplainerHub(model, x)
    assert hub.shap_algorithm == "TreeSHAP"

    shap_exp = hub.explain_shap(x.iloc[0])
    assert len(shap_exp["values"]) == len(FEATURE_NAMES)
    assert shap_exp["latency_s"] > 0

    lime_exp = hub.explain_lime(x.iloc[0], num_features=5)
    assert len(lime_exp["contributions"]) == 5
    assert lime_exp["latency_s"] > 0

    top = hub.global_importance(x.head(50))
    # The label was built from feature 0, so it must dominate importance.
    assert top.iloc[0]["feature"] == FEATURE_NAMES[0]


@pytest.mark.skipif(not (DATA_DIR / "X_test.csv").exists(), reason="real dataset not available")
def test_load_real_test_data():
    x, y = load_test_data(n_rows=100)
    assert list(x.columns) == FEATURE_NAMES
    assert set(y.unique()) <= {0, 1}
