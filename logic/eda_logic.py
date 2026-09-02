"""
eda_logic.py — All logic for SQL querying and EDA.
No Streamlit imports.
"""
import io
import re
import pandas as pd
from .llm_client import ask


def text_to_sql(user_question: str, schema: str) -> str:
    prompt = f"""
You are an expert MySQL analyst for a banking system.
Given this database schema:
{schema}

Write a valid MySQL SELECT query that answers:
"{user_question}"

Rules:
- Return ONLY the raw SQL. No markdown. No explanation.
- Use JOINs when data spans multiple tables.
- Limit results to 100 rows unless asked otherwise.
"""
    raw = ask(prompt, temperature=0.1)
    return re.sub(r"```(?:sql)?|```", "", raw).strip()


def deep_eda_report(df: pd.DataFrame) -> str:
    numeric_summary  = df.describe().to_string()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    cat_summary      = {c: df[c].value_counts().head(5).to_dict() for c in categorical_cols}
    missing          = df.isnull().sum().to_dict()

    prompt = f"""
You are a Senior Data Scientist at a BFSI firm.
Perform a thorough EDA on the dataset described below and format the output as clean Markdown.

Shape: {df.shape[0]} rows x {df.shape[1]} columns
Missing values: {missing}
Numeric summary:
{numeric_summary}
Categorical top-5 distributions: {cat_summary}

Provide:
1. Dataset Overview
2. Univariate Summary (key observations per column)
3. Multivariate Patterns and Correlations
4. Data Quality Issues
5. Recommended Cleaning Steps
6. Feature Engineering Suggestions (banking-relevant)
"""
    return ask(prompt)


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_sql_insert(df: pd.DataFrame, table_name: str) -> str:
    rows = []
    for _, row in df.iterrows():
        vals = ", ".join(f"'{str(v).replace(chr(39), chr(39)*2)}'" for v in row)
        rows.append(f"INSERT INTO `{table_name}` VALUES ({vals});")
    return "\n".join(rows)
