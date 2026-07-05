"""Model performance metrics and inference latency benchmarking."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, x: pd.DataFrame, y: pd.Series) -> dict:
    """Score a fitted classifier on (x, y) and measure batch inference latency.

    Returns accuracy/precision/recall/F1 (binary, attack = positive class),
    the confusion matrix, and wall-clock latency both for the full batch
    and per sample (microseconds) — the number a SOC cares about when
    sizing an inline detector.
    """
    start = time.perf_counter()
    y_pred = model.predict(x.values)
    elapsed = time.perf_counter() - start

    return {
        "n_samples": len(y),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]),
        "latency_batch_s": elapsed,
        "latency_per_sample_us": elapsed / max(len(y), 1) * 1e6,
    }


def predict_one(model, row: pd.Series) -> dict:
    """Classify a single flow/packet row; return label, probability, latency."""
    x = row.to_frame().T.values.astype(np.float64)
    start = time.perf_counter()
    proba = model.predict_proba(x)[0]
    elapsed = time.perf_counter() - start
    label = int(np.argmax(proba))
    return {
        "label": label,
        "proba_benign": float(proba[0]),
        "proba_attack": float(proba[1]),
        "latency_ms": elapsed * 1e3,
    }
