"""
db_core.py — Centralised MySQL connection + schema management
Reads all credentials from .env via python-dotenv.
No Streamlit imports.
"""
import os
import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from dotenv import load_dotenv

load_dotenv()

_CONFIG = {
    "host":     os.getenv("MYSQL_HOST",     "localhost"),
    "user":     os.getenv("MYSQL_USER",     "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB",       "finpy"),
}


def get_connection():
    try:
        return mysql.connector.connect(**_CONFIG)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            base = {k: v for k, v in _CONFIG.items() if k != "database"}
            conn = mysql.connector.connect(**base)
            cur = conn.cursor()
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{_CONFIG['database']}`")
            conn.commit()
            cur.close()
            conn.close()
            return mysql.connector.connect(**_CONFIG)
        raise


def init_schema():
    """Create the 3-table banking schema if it does not exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id       INT AUTO_INCREMENT PRIMARY KEY,
            full_name         VARCHAR(150),
            phone_number      VARCHAR(15) UNIQUE,
            age               INT,
            annual_income     FLOAT,
            credit_score      INT,
            employment_years  INT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS loans (
            loan_id                  INT AUTO_INCREMENT PRIMARY KEY,
            customer_id              INT,
            total_existing_debt      FLOAT,
            loan_amount_requested    FLOAT,
            loan_tenure_months       INT,
            interest_rate            FLOAT,
            default_status           TINYINT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id          INT AUTO_INCREMENT PRIMARY KEY,
            account_number          VARCHAR(20),
            customer_id             INT,
            transaction_amount      FLOAT,
            hour_of_day             INT,
            distance_from_home_km   FLOAT,
            transactions_last_24h   INT,
            avg_txn_amount_30d      FLOAT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """)

    conn.commit()
    cur.close()
    conn.close()


def query_to_df(sql: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def get_schema() -> str:
    """Return human-readable schema for all 3 tables."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SHOW TABLES")
    tables = [r[0] for r in cur.fetchall()]
    lines = []
    for t in tables:
        cur.execute(f"DESCRIBE `{t}`")
        cols = ", ".join(f"{c[0]} ({c[1]})" for c in cur.fetchall())
        lines.append(f"Table: {t}\n  Columns: {cols}")
    cur.close()
    conn.close()
    return "\n\n".join(lines) if lines else "No tables found."
