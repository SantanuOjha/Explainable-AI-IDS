"""Interactive XAI-IDS evaluation dashboard.

Run locally:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when launched via `streamlit run app/dashboard.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import components  # noqa: E402
from src.evaluation import evaluate_model, predict_one  # noqa: E402
from src.explainers import ExplainerHub  # noqa: E402
from src.utils import available_models, load_model, load_test_data  # noqa: E402

st.set_page_config(page_title="XAI-IDS Dashboard", page_icon="🛡️", layout="wide")

# Rows used for the (cached) global SHAP pass and the metric evaluation.
GLOBAL_SAMPLE = 300
EVAL_SAMPLE = 5000


# --------------------------------------------------------------- cached layer
@st.cache_data(show_spinner="Loading held-out test data …")
def get_test_data():
    return load_test_data()


@st.cache_resource(show_spinner="Loading model + building explainers …")
def get_model_and_hub(model_name: str):
    """One model + one ExplainerHub per selection, built once per session.

    The hub caches the SHAP tree structures and the LIME background
    statistics, so per-packet explanations only pay per-row cost.
    """
    model = load_model(available_models()[model_name])
    x, _ = get_test_data()
    return model, ExplainerHub(model, x.sample(min(len(x), 2000), random_state=0))


@st.cache_data(show_spinner="Evaluating model on the test split …")
def get_metrics(model_name: str):
    model, _ = get_model_and_hub(model_name)
    x, y = get_test_data()
    return evaluate_model(model, x.head(EVAL_SAMPLE), y.head(EVAL_SAMPLE))


@st.cache_data(show_spinner="Computing global SHAP importance …")
def get_global_importance(model_name: str) -> pd.DataFrame:
    _, hub = get_model_and_hub(model_name)
    x, _ = get_test_data()
    return hub.global_importance(x.sample(GLOBAL_SAMPLE, random_state=0))


# ------------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ XAI-IDS")
st.sidebar.caption("Explainable intrusion detection on CICIDS2017")

models = available_models()
if not models:
    st.error(
        "No trained models found under `Data/Sampled_Data/`. "
        "Run the training notebooks (03/04) or mount the Data directory."
    )
    st.stop()

model_name = st.sidebar.selectbox("Model", list(models))
model, hub = get_model_and_hub(model_name)
X_test, y_test = get_test_data()

st.sidebar.markdown(f"**SHAP algorithm:** {hub.shap_algorithm}")
st.sidebar.markdown(f"**Test flows:** {len(X_test):,}")
st.sidebar.markdown(f"**Attack rate:** {y_test.mean():.1%}")

# ---------------------------------------------------------------------- tabs
tab_global, tab_inspect = st.tabs(["📊 Global Overview", "🔎 Packet Inspector"])

with tab_global:
    st.subheader(f"Model performance — {model_name}")
    st.caption(f"Evaluated on the first {EVAL_SAMPLE:,} held-out test flows.")
    metrics = get_metrics(model_name)
    components.metrics_row(metrics)

    col_cm, col_imp = st.columns([1, 2])
    with col_cm:
        st.markdown("**Confusion matrix**")
        components.confusion_matrix_table(metrics)
    with col_imp:
        components.global_importance_chart(get_global_importance(model_name))
        st.caption(
            f"Mean |SHAP| over {GLOBAL_SAMPLE} sampled flows — what the model "
            "prioritizes across the whole dataset, not just one alert."
        )

with tab_inspect:
    st.subheader("Inspect a single network flow")

    pick_col, edit_col = st.columns([1, 1])
    with pick_col:
        pool = st.radio(
            "Sample pool",
            ["Any", "True ATTACK flows", "True BENIGN flows"],
            horizontal=True,
        )
        if pool == "True ATTACK flows":
            candidates = y_test[y_test == 1].index
        elif pool == "True BENIGN flows":
            candidates = y_test[y_test == 0].index
        else:
            candidates = y_test.index
        idx = st.selectbox(
            "Flow index (held-out test set)",
            candidates[:5000],
            help="First 5,000 matching flows are listed.",
        )

    row = X_test.loc[idx].copy()

    with edit_col:
        with st.expander("✏️ Manually override feature values"):
            st.caption(
                "Values are in standardized units (the model's input space), "
                "since the training scaler operates on z-scores."
            )
            top_features = get_global_importance(model_name)["feature"].head(8)
            for feat in top_features:
                row[feat] = st.number_input(
                    feat, value=float(row[feat]), format="%.4f", key=f"edit_{feat}"
                )

    st.markdown(f"**Ground-truth label:** {'🚨 ATTACK' if y_test.loc[idx] == 1 else '✅ BENIGN'}")
    components.prediction_banner(predict_one(model, row))

    st.divider()
    st.subheader("Why did the model decide this?")

    with st.spinner("Generating SHAP + LIME explanations …"):
        shap_exp = hub.explain_shap(row)
        lime_exp = hub.explain_lime(row)

    view = st.radio("Explanation view", ["Side by side", "SHAP only", "LIME only"], horizontal=True)
    if view == "Side by side":
        col_shap, col_lime = st.columns(2)
        with col_shap:
            st.markdown(f"**SHAP — {shap_exp['algorithm']}**")
            components.shap_waterfall(shap_exp)
        with col_lime:
            st.markdown(f"**{lime_exp['algorithm']}**")
            components.lime_bar_chart(lime_exp)
    elif view == "SHAP only":
        components.shap_waterfall(shap_exp, max_display=16)
    else:
        components.lime_bar_chart(lime_exp)

    st.divider()
    st.subheader("Explainer performance")
    components.latency_comparison(shap_exp["latency_s"], lime_exp["latency_s"])
