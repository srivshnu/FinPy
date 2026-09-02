"""
page1_eda.py — Agentic Data Analyst
UI layer only. All logic delegated to logic/ modules.
"""
import io
import sys, os
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from logic.eda_logic   import (deep_eda_report, text_to_sql, text_to_pandas,
                                df_to_excel_bytes, df_to_csv_bytes, df_to_sql_insert)
from logic.plot_logic  import correlation_heatmap, distribution_grid, missing_bar
from logic.db_core     import query_to_df, get_schema


def _load_df(source_key="p1_df") -> pd.DataFrame | None:
    return st.session_state.get(source_key)


def render():
    st.header("🔍 Agentic Data Analyst")
    st.markdown("Upload a CSV / Excel file **or** connect to your MySQL database — then ask questions in plain English.")

    # ── Data Source ──────────────────────────────────────────────────────────
    src = st.radio("Choose data source", ["Upload File", "MySQL Database"], horizontal=True)

    if src == "Upload File":
        file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if file:
            df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            st.session_state["p1_df"]   = df
            st.session_state["p1_mode"] = "pandas"
            st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    else:
        query_input = st.text_input("Enter a SELECT query to load data from MySQL",
                                    placeholder="SELECT * FROM credit_risk_data LIMIT 500")
        if st.button("Load from MySQL") and query_input:
            try:
                df = query_to_df(query_input)
                st.session_state["p1_df"]   = df
                st.session_state["p1_mode"] = "sql"
                st.success(f"Loaded {df.shape[0]:,} rows")
            except Exception as e:
                st.error(f"Database error: {e}")

    df = _load_df()
    if df is None:
        st.info("No data loaded yet. Upload a file or run a query above.")
        return

    st.divider()

    # ── Preview ───────────────────────────────────────────────────────────────
    with st.expander("📄 Raw Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # ── Visualisations ────────────────────────────────────────────────────────
    st.subheader("📊 Automated Visualisations")
    c1, c2, c3 = st.columns(3)

    if c1.button("Distribution Grid"):
        st.pyplot(distribution_grid(df))

    if c2.button("Correlation Heatmap"):
        st.pyplot(correlation_heatmap(df))

    if c3.button("Missing Values"):
        st.pyplot(missing_bar(df))

    st.divider()

    # ── Deep AI EDA ───────────────────────────────────────────────────────────
    st.subheader("🤖 Deep AI EDA Report")
    if st.button("Generate AI EDA Report"):
        with st.spinner("LLM is analysing your dataset..."):
            try:
                report = deep_eda_report(df)
                st.markdown(report)
            except Exception as e:
                st.error(str(e))

    st.divider()

    # ── Chat Interface ────────────────────────────────────────────────────────
    st.subheader("💬 Ask a Question About Your Data")
    mode = st.radio("Query mode", ["Text-to-Pandas (CSV/Excel)", "Text-to-SQL (MySQL)"], horizontal=True)
    question = st.text_input("Ask a question in plain English",
                              placeholder="e.g. What is the average income by credit score band?")

    if st.button("Run Query") and question:
        with st.spinner("LLM generating query..."):
            try:
                if mode.startswith("Text-to-SQL"):
                    schema = get_schema()
                    sql = text_to_sql(question, schema)
                    st.code(sql, language="sql")
                    result = query_to_df(sql)
                else:
                    code = text_to_pandas(question, list(df.columns), df.head(3).to_string())
                    st.code(code, language="python")
                    local_env = {"df": df, "pd": pd}
                    exec(code, local_env)
                    result = local_env.get("result", "No `result` variable found.")

                if isinstance(result, pd.DataFrame):
                    st.dataframe(result, use_container_width=True)
                    st.session_state["p1_df"] = result   # update for downstream use
                else:
                    st.write(result)
            except Exception as e:
                st.error(str(e))

    st.divider()

    # ── Format Conversion ─────────────────────────────────────────────────────
    st.subheader("🔄 Export / Convert")
    e1, e2, e3 = st.columns(3)

    e1.download_button("⬇ Download CSV",
        data=df_to_csv_bytes(df), file_name="export.csv", mime="text/csv")

    e2.download_button("⬇ Download Excel",
        data=df_to_excel_bytes(df), file_name="export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    table_name = st.text_input("Table name for SQL INSERT export", value="my_table")
    e3.download_button("⬇ Download SQL INSERTs",
        data=df_to_sql_insert(df, table_name).encode(), file_name="export.sql", mime="text/plain")
