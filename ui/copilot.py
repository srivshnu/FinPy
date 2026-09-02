"""
copilot.py — The Unified Agentic Banking Copilot
Single-page chat interface. User types a scenario in plain English,
selects an action, and the AI agent handles everything:
  - Parses text into structured data (LLM)
  - Anonymises PII (Hasher)
  - Runs inference (Pre-trained models)
  - Or queries the database (Text-to-SQL)
"""
import sys, os
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from logic.inference_logic import (
    parse_credit_from_text, run_credit_inference,
    parse_anomaly_from_text, run_anomaly_inference,
    CREDIT_FEATURES, ANOMALY_FEATURES
)
from logic.eda_logic  import text_to_sql, deep_eda_report, df_to_csv_bytes, df_to_excel_bytes
from logic.db_core    import query_to_df, get_schema
from logic.plot_logic import anomaly_scatter, correlation_heatmap

import numpy as np


def _init_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Welcome to the **FinPy Banking Copilot**.\n\n"
                    "Describe a customer scenario, a transaction, or ask a database question "
                    "in plain English — then choose what you want me to do with it.\n\n"
                    "**Examples:**\n"
                    "- *A 34-year-old with ₹8L annual income, credit score 620, wants ₹3L loan, existing debt ₹80K*\n"
                    "- *Account 401928374651 transferred ₹4,50,000 at 3 AM, 4500 km from home branch*\n"
                    "- *What is the average loan amount grouped by credit score band?*"
                )
            }
        ]


def _render_credit_result(result: dict):
    prob = result["default_probability_pct"]
    verdict = result["verdict"]

    if "HIGH" in verdict:
        st.error(f"🚨 **{verdict}** — Default Probability: {prob:.1f}%")
    elif "MODERATE" in verdict:
        st.warning(f"⚠️ **{verdict}** — Default Probability: {prob:.1f}%")
    else:
        st.success(f"✅ **{verdict}** — Default Probability: {prob:.1f}%")

    st.progress(int(prob), text=f"Default Probability: {prob:.1f}%")
    st.info(f"**Recommendation:** {result['recommendation']}")

    col1, col2 = st.columns(2)
    col1.metric("Anonymised User ID", result["user_hash"][:16] + "…")
    col2.metric("Default Probability", f"{prob:.2f}%")

    with st.expander("📋 Feature Vector (After PII Removal)"):
        st.dataframe(
            pd.DataFrame({"Feature": list(result["features"].keys()),
                          "Value": list(result["features"].values())}),
            use_container_width=True, hide_index=True
        )


def _render_anomaly_result(result: dict):
    dist     = result["centroid_distance"]
    thresh   = result["threshold"]
    verdict  = result["verdict"]

    if result["is_anomaly"]:
        st.error(f"🚨 **{verdict}**")
    else:
        st.success(f"✅ **{verdict}**")

    st.info(f"**Recommendation:** {result['recommendation']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Centroid Distance", f"{dist:.4f}")
    col2.metric("Threshold", f"{thresh:.4f}")
    col3.metric("Anomised Txn ID", result["txn_hash"][:12] + "…")

    excess_pct = min(100, int((dist / (thresh * 1.5)) * 100))
    st.progress(excess_pct, text=f"Distance vs 1.5× threshold: {excess_pct}%")

    # Scatter plot — simulate cluster context
    np.random.seed(1)
    sim_df = pd.DataFrame({
        "transaction_amount":    np.random.uniform(50, 5000, 300),
        "distance_from_home_km": np.random.uniform(0, 50, 300),
        "is_anomaly":            0
    })
    new_pt = pd.DataFrame({
        "transaction_amount":    [result["features"]["transaction_amount"]],
        "distance_from_home_km": [result["features"]["distance_from_home_km"]],
        "is_anomaly":            [1 if result["is_anomaly"] else 0]
    })
    plot_df = pd.concat([sim_df, new_pt], ignore_index=True)
    st.pyplot(anomaly_scatter(plot_df, "transaction_amount", "distance_from_home_km"))

    with st.expander("📋 Feature Vector (After PII Removal)"):
        st.dataframe(
            pd.DataFrame({"Feature": list(result["features"].keys()),
                          "Value": list(result["features"].values())}),
            use_container_width=True, hide_index=True
        )


def _render_sql_result(df: pd.DataFrame, sql: str):
    st.code(sql, language="sql")
    if df.empty:
        st.warning("Query returned 0 rows.")
        return
    st.success(f"Returned {len(df):,} rows.")
    st.dataframe(df, use_container_width=True)

    # Show correlation heatmap if enough numeric columns
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) >= 2:
        with st.expander("📊 Correlation Heatmap"):
            st.pyplot(correlation_heatmap(df))

    # Export options
    c1, c2 = st.columns(2)
    c1.download_button("⬇ CSV", df_to_csv_bytes(df), "result.csv", "text/csv")
    c2.download_button("⬇ Excel", df_to_excel_bytes(df), "result.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def render():
    st.title("🏦 FinPy Banking Copilot")
    st.caption("Powered by XGBoost · K-Means · Gemini LLM · MySQL")
    st.divider()

    _init_chat()

    # ── Chat History ─────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input Area ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🎯 New Query")

    user_input = st.text_area(
        "Describe your scenario or question in plain English",
        placeholder=(
            "e.g. A 42-year-old doctor, phone 9123456789, earns ₹18 lakh/year, "
            "credit score 720, wants ₹10 lakh home loan for 120 months at 9.5%, "
            "existing debt ₹2.5 lakh."
        ),
        height=110,
        key="user_input_area"
    )

    action = st.radio(
        "What should the AI do with this?",
        [
            "📊 Credit Risk Evaluation",
            "🚨 Fraud / Anomaly Detection",
            "🔍 Database SQL Query",
        ],
        horizontal=True
    )

    run_btn = st.button("🚀 Run Copilot", type="primary", use_container_width=True)

    if run_btn:
        if not user_input.strip():
            st.warning("Please type a scenario or question above.")
            return

        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # ── CREDIT RISK ───────────────────────────────────────────────
            if action.startswith("📊"):
                with st.spinner("Parsing scenario with LLM..."):
                    try:
                        parsed = parse_credit_from_text(user_input)
                    except Exception as e:
                        st.error(f"LLM parsing error: {e}")
                        return

                with st.expander("🔍 Parsed by LLM (before hashing)", expanded=False):
                    import json
                    st.json(parsed)

                with st.spinner("Hashing PII and running XGBoost inference..."):
                    try:
                        result = run_credit_inference(parsed)
                        _render_credit_result(result)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": (
                                f"**Credit Risk Result:** {result['verdict']} "
                                f"({result['default_probability_pct']:.1f}% default probability)\n\n"
                                f"_{result['recommendation']}_"
                            )
                        })
                    except FileNotFoundError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Inference error: {e}")

            # ── ANOMALY DETECTION ─────────────────────────────────────────
            elif action.startswith("🚨"):
                with st.spinner("Parsing transaction with LLM..."):
                    try:
                        parsed = parse_anomaly_from_text(user_input)
                    except Exception as e:
                        st.error(f"LLM parsing error: {e}")
                        return

                with st.expander("🔍 Parsed by LLM (before hashing)", expanded=False):
                    import json
                    st.json(parsed)

                with st.spinner("Hashing PII and running K-Means scoring..."):
                    try:
                        result = run_anomaly_inference(parsed)
                        _render_anomaly_result(result)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": (
                                f"**Fraud Detection Result:** {result['verdict']} "
                                f"(Distance: {result['centroid_distance']:.4f} / Threshold: {result['threshold']:.4f})\n\n"
                                f"_{result['recommendation']}_"
                            )
                        })
                    except FileNotFoundError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Inference error: {e}")

            # ── TEXT-TO-SQL ───────────────────────────────────────────────
            else:
                with st.spinner("LLM generating SQL query..."):
                    try:
                        schema = get_schema()
                        sql    = text_to_sql(user_input, schema)
                        df     = query_to_df(sql)
                        _render_sql_result(df, sql)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": (
                                f"**SQL Query executed successfully.**\n"
                                f"```sql\n{sql}\n```\n"
                                f"Returned {len(df):,} rows."
                            )
                        })
                    except Exception as e:
                        st.error(f"SQL error: {e}")

    # ── Schema Reference (collapsed sidebar) ────────────────────────────────
    with st.sidebar:
        st.markdown("### 🗃 Database Schema")
        try:
            st.text(get_schema())
        except Exception:
            st.caption("Connect to MySQL to view schema.")
        st.divider()
        if st.button("🗑 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
