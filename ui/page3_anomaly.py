"""
page3_anomaly.py — Fraud / Anomaly Detection Inference Interface
UI layer only. All logic delegated to logic/ modules.
"""
import sys, os
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from logic.inference_logic import (
    generate_anomaly_sample, run_anomaly_inference, ANOMALY_SCHEMA, ANOMALY_FEATURE_COLS
)
from logic.plot_logic import anomaly_scatter


def render():
    st.header("🚨 Fraud / Anomaly Detection Inference")
    st.markdown(
        "A pre-trained **K-Means** model detects whether a banking transaction is "
        "anomalous (likely fraudulent) by measuring how far it sits from known "
        "normal transaction clusters.  \n"
        "All **Personally Identifiable Information (PII)** is hashed using the "
        "customer's account number before the model ever sees the data."
    )

    st.info(
        "**Model:** K-Means Clustering (k=3, distance threshold = 95th percentile)  |  "
        "**Training dataset:** 1,900 normal + 100 simulated fraud transactions  |  "
        "**Features:** " + ", ".join(ANOMALY_FEATURE_COLS),
        icon="ℹ️"
    )

    st.divider()

    # ── Test Mode ─────────────────────────────────────────────────────────────
    st.subheader("🧪 Run an Inference Test")
    mode = st.radio(
        "How would you like to provide transaction data?",
        ["LLM Auto-Generate (1 sample)", "Type in Plain English"],
        horizontal=True
    )

    user_text = ""
    if mode == "Type in Plain English":
        user_text = st.text_area(
            "Describe the transaction in plain English",
            placeholder="e.g. A transaction of ₹3,00,000 made at 2 AM, "
                        "5000 km from the home branch, account had 45 transactions in last 24 hours.",
            height=100
        )

    if st.button("🚀 Generate Sample & Detect Anomaly", type="primary"):
        with st.spinner("LLM generating transaction sample..."):
            try:
                sample = generate_anomaly_sample(user_text)
            except Exception as e:
                st.error(f"LLM Error: {e}")
                return

        st.success("Sample generated. Hashing PII and running anomaly detection...")

        with st.expander("🔍 Raw LLM-Generated Sample (before hashing)", expanded=True):
            st.json(sample)

        try:
            result = run_anomaly_inference(sample)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Run `python scripts/train_anomaly_model.py` from the project root to generate the model.")
            return
        except Exception as e:
            st.error(f"Inference Error: {e}")
            return

        st.divider()

        # ── Results ────────────────────────────────────────────────────────
        st.subheader("🔐 After PII Hashing — What the Model Sees")
        col_hash, col_dist = st.columns([2, 1])
        col_hash.metric("Anonymised Transaction Hash", result["txn_hash"][:20] + "...")
        col_dist.metric("Centroid Distance", f"{result['centroid_distance']:.4f}")

        if "ANOMALY" in result["verdict"]:
            st.error(result["verdict"])
        else:
            st.success(result["verdict"])

        # Distance vs threshold gauge
        dist  = result["centroid_distance"]
        thresh = result["threshold"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Distance",  f"{dist:.4f}")
        col2.metric("Threshold", f"{thresh:.4f}")
        col3.metric("Excess",    f"{max(0, dist - thresh):.4f}")

        pct = min(100, int((dist / (thresh * 1.5)) * 100))
        st.progress(pct, text=f"Distance as % of 1.5× threshold: {pct}%")

        # Feature table
        st.subheader("📋 Feature Vector Passed to Model")
        st.dataframe(
            {"Feature": list(result["features_used"].keys()),
             "Value":   list(result["features_used"].values())},
            use_container_width=True
        )

        # Mini scatter simulation
        st.subheader("📈 Anomaly Context (Simulated Cluster View)")
        st.caption("Simulated normal cluster + this transaction plotted in 2D for visualisation.")
        np.random.seed(1)
        sim_df = pd.DataFrame({
            "transaction_amount":    np.random.uniform(50, 5000, 200),
            "distance_from_home_km": np.random.uniform(0, 50, 200),
            "is_anomaly":            0
        })
        test_row = pd.DataFrame({
            "transaction_amount":    [result["features_used"]["transaction_amount"]],
            "distance_from_home_km": [result["features_used"]["distance_from_home_km"]],
            "is_anomaly":            [1]
        })
        plot_df = pd.concat([sim_df, test_row], ignore_index=True)
        st.pyplot(anomaly_scatter(plot_df, "transaction_amount", "distance_from_home_km"))

    st.divider()

    with st.expander("📖 Data Schema used for Training"):
        for k, v in ANOMALY_SCHEMA.items():
            st.markdown(f"- **`{k}`**: {v}")
