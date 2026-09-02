"""
page2_credit_risk.py — Credit Risk Inference Interface
UI layer only. All logic delegated to logic/ modules.
"""
import sys, os
import json
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from logic.inference_logic import (
    generate_credit_sample, run_credit_inference, CREDIT_SCHEMA, CREDIT_FEATURE_COLS
)
from logic.plot_logic import roc_curve_plot


def render():
    st.header("📊 Credit Risk Inference")
    st.markdown(
        "A pre-trained **XGBoost** model predicts the probability that a customer "
        "will default on a loan.  \n"
        "All **Personally Identifiable Information (PII)** is hashed using the customer's "
        "phone number before the feature vector reaches the model — engineers never see raw PII."
    )

    st.info(
        "**Model:** XGBoost Classifier  |  "
        "**Training dataset:** 2,000 anonymised synthetic banking profiles  |  "
        "**Features used:** " + ", ".join(CREDIT_FEATURE_COLS),
        icon="ℹ️"
    )

    st.divider()

    # ── Test Mode ─────────────────────────────────────────────────────────────
    st.subheader("🧪 Run an Inference Test")
    mode = st.radio(
        "How would you like to provide customer data?",
        ["LLM Auto-Generate (1 sample)", "Type in Plain English"],
        horizontal=True
    )

    user_text = ""
    if mode == "Type in Plain English":
        user_text = st.text_area(
            "Describe the customer in plain English",
            placeholder="e.g. A 35 year old customer with a 680 credit score, annual income of ₹6 lakh, "
                        "requesting a ₹2 lakh loan for 36 months with existing debt of ₹80,000.",
            height=100
        )

    if st.button("🚀 Generate Sample & Run Prediction", type="primary"):
        with st.spinner("LLM generating sample..."):
            try:
                sample = generate_credit_sample(user_text)
            except Exception as e:
                st.error(f"LLM Error: {e}")
                return

        st.success("Sample generated. Hashing PII and running inference...")

        # Show raw sample before hashing
        with st.expander("🔍 Raw LLM-Generated Sample (before hashing)", expanded=True):
            st.json(sample)

        try:
            result = run_credit_inference(sample)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Run `python scripts/train_credit_model.py` from the project root to generate the model.")
            return
        except Exception as e:
            st.error(f"Inference Error: {e}")
            return

        st.divider()

        # ── Results ────────────────────────────────────────────────────────
        st.subheader("🔐 After PII Hashing — What the Model Sees")
        col_hash, col_verdict = st.columns([2, 1])
        col_hash.metric("Anonymised User Hash", result["user_hash"][:20] + "...")
        col_verdict.metric("Verdict", result["verdict"][:10])

        # Big verdict banner
        if "HIGH" in result["verdict"]:
            st.error(f"🚨  {result['verdict']}")
        else:
            st.success(f"✅  {result['verdict']}")

        # Probability gauge
        prob = result["default_probability"]
        col1, col2 = st.columns(2)
        col1.metric("Default Probability", f"{prob:.2f}%")
        col2.progress(int(prob))

        # Feature table
        st.subheader("📋 Feature Vector Passed to Model")
        st.dataframe(
            {"Feature": list(result["features_used"].keys()),
             "Value":   list(result["features_used"].values())},
            use_container_width=True
        )

    st.divider()

    # ── Model Info ─────────────────────────────────────────────────────────────
    with st.expander("📖 Data Schema used for Training"):
        for k, v in CREDIT_SCHEMA.items():
            st.markdown(f"- **`{k}`**: {v}")
