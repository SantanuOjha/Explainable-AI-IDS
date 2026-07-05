"""Reusable Streamlit UI components for the XAI-IDS dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

_BENIGN_COLOR = "#2e7d32"
_ATTACK_COLOR = "#c62828"


def metrics_row(metrics: dict) -> None:
    """Render headline model metrics as Streamlit metric cards."""
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
    cols[1].metric("Precision", f"{metrics['precision']:.4f}")
    cols[2].metric("Recall", f"{metrics['recall']:.4f}")
    cols[3].metric("F1-score", f"{metrics['f1']:.4f}")
    cols[4].metric("Latency / flow", f"{metrics['latency_per_sample_us']:.1f} µs")


def confusion_matrix_table(metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual BENIGN", "Actual ATTACK"],
            columns=["Predicted BENIGN", "Predicted ATTACK"],
        ),
        width="stretch",
    )


def global_importance_chart(importance: pd.DataFrame, top_n: int = 15) -> None:
    """Horizontal bar chart of mean |SHAP| global feature importance."""
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * top_n + 1))
    ax.barh(top["feature"], top["importance"], color="#1565c0")
    ax.set_xlabel("mean |SHAP value|  (impact on ATTACK score)")
    ax.set_title(f"Top {top_n} features the model relies on")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def prediction_banner(pred: dict) -> None:
    """Verdict banner + probability gauge for a single inspected flow."""
    if pred["label"] == 1:
        st.error(
            f"🚨 **ATTACK detected** — confidence {pred['proba_attack']:.1%} "
            f"(inference {pred['latency_ms']:.2f} ms)"
        )
    else:
        st.success(
            f"✅ **Benign traffic** — confidence {pred['proba_benign']:.1%} "
            f"(inference {pred['latency_ms']:.2f} ms)"
        )
    st.progress(pred["proba_attack"], text=f"P(ATTACK) = {pred['proba_attack']:.4f}")


def shap_waterfall(explanation: dict, max_display: int = 12) -> None:
    """Local SHAP waterfall: which features pushed this flow toward ATTACK."""
    exp = shap.Explanation(
        values=explanation["values"],
        base_values=explanation["base_value"],
        data=explanation["data"],
        feature_names=explanation["features"],
    )
    fig = plt.figure()
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def lime_bar_chart(explanation: dict) -> None:
    """Local LIME contributions; red pushes toward ATTACK, green toward BENIGN."""
    pairs = explanation["contributions"][::-1]
    labels = [p[0] for p in pairs]
    weights = np.array([p[1] for p in pairs])
    colors = [_ATTACK_COLOR if w > 0 else _BENIGN_COLOR for w in weights]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(pairs) + 1))
    ax.barh(labels, weights, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight  (→ ATTACK is positive)")
    ax.set_title("LIME local surrogate explanation")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def latency_comparison(shap_s: float, lime_s: float) -> None:
    """Side-by-side execution-time widget for SHAP vs LIME."""
    col1, col2, col3 = st.columns(3)
    col1.metric("SHAP latency", f"{shap_s * 1e3:.1f} ms")
    col2.metric("LIME latency", f"{lime_s * 1e3:.1f} ms")
    faster = "SHAP" if shap_s < lime_s else "LIME"
    ratio = max(shap_s, lime_s) / max(min(shap_s, lime_s), 1e-9)
    col3.metric("Faster explainer", faster, f"{ratio:.1f}× speedup")

    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.barh(["LIME", "SHAP"], [lime_s * 1e3, shap_s * 1e3], color=["#6a1b9a", "#1565c0"])
    ax.set_xlabel("explanation time (ms)")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
