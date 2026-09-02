"""
inference_logic.py
The Agentic Brain:
  1. LLM parses free-form natural language into a structured JSON record.
  2. Hasher anonymises PII.
  3. Pre-trained model runs inference and returns a structured result dict.

No Streamlit imports.
"""
import re
import json
import joblib
import numpy as np
from pathlib import Path
from .llm_client import ask
from .hasher import hash_identity

MODELS_DIR = Path(__file__).parent.parent / "models"

# ── Fixed feature schemas ─────────────────────────────────────────────────────
CREDIT_FEATURES = [
    "age", "annual_income", "credit_score", "employment_years",
    "total_existing_debt", "loan_amount_requested",
    "loan_tenure_months", "interest_rate"
]

ANOMALY_FEATURES = [
    "transaction_amount", "hour_of_day", "distance_from_home_km",
    "transactions_last_24h", "avg_txn_amount_30d"
]


# ── LLM Parser ────────────────────────────────────────────────────────────────

def parse_credit_from_text(user_text: str) -> dict:
    """Extract a credit risk feature record from free-form text using the LLM."""
    prompt = f"""
You are a banking data extraction engine.
Extract the following fields from the user's description and return ONLY a valid JSON object.
If a field is not mentioned, infer a realistic default value for a typical banking customer.

Required fields:
{{
  "phone_number": "10-digit phone number string, use '9999999999' if not mentioned",
  "full_name": "customer name string, use 'Unknown' if not mentioned",
  "age": integer (18-75),
  "annual_income": float in INR (yearly),
  "credit_score": integer (300-900),
  "employment_years": integer (0-40),
  "total_existing_debt": float in INR,
  "loan_amount_requested": float in INR,
  "loan_tenure_months": integer (12-360),
  "interest_rate": float percentage (7.0-24.0)
}}

User description:
\"\"\"{user_text}\"\"\"

Return ONLY the JSON. No markdown. No explanation.
"""
    raw = ask(prompt, temperature=0.1)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    return json.loads(raw)


def parse_anomaly_from_text(user_text: str) -> dict:
    """Extract a transaction anomaly feature record from free-form text using the LLM."""
    prompt = f"""
You are a banking fraud detection data extraction engine.
Extract the following fields from the user's transaction description and return ONLY valid JSON.
If a field is not mentioned, infer a realistic default for a normal banking transaction.

Required fields:
{{
  "account_number": "12-digit account number string, use '000000000000' if not mentioned",
  "account_holder": "account holder name string, use 'Unknown' if not mentioned",
  "transaction_amount": float in INR,
  "hour_of_day": integer (0-23),
  "distance_from_home_km": float (how far the transaction is from home branch in km),
  "transactions_last_24h": integer (number of transactions in last 24 hours),
  "avg_txn_amount_30d": float in INR (customer's 30-day average transaction amount)
}}

User description:
\"\"\"{user_text}\"\"\"

Return ONLY the JSON. No markdown. No explanation.
"""
    raw = ask(prompt, temperature=0.1)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    return json.loads(raw)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_credit_inference(parsed: dict) -> dict:
    """Hash PII, run XGBoost, return verdict dict."""
    model_path = MODELS_DIR / "credit_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("credit_model.pkl not found. Run scripts/train_credit_model.py first.")

    user_hash = hash_identity(
        str(parsed.get("phone_number", "unknown")),
        str(parsed.get("full_name", ""))
    )

    features = np.array([[parsed[c] for c in CREDIT_FEATURES]], dtype=float)
    model = joblib.load(model_path)
    prob  = model.predict_proba(features)[0][1]

    if prob >= 0.7:
        verdict = "HIGH RISK"
        recommendation = "Loan application should be flagged for manual review. Collateral or guarantor recommended."
    elif prob >= 0.4:
        verdict = "MODERATE RISK"
        recommendation = "Conditionally approve with higher interest rate or reduced loan amount."
    else:
        verdict = "LOW RISK"
        recommendation = "Eligible for standard loan processing."

    return {
        "action": "credit_risk",
        "user_hash": user_hash,
        "default_probability_pct": round(prob * 100, 2),
        "verdict": verdict,
        "recommendation": recommendation,
        "features": dict(zip(CREDIT_FEATURES, features[0].tolist()))
    }


def run_anomaly_inference(parsed: dict) -> dict:
    """Hash PII, run K-Means distance scoring, return verdict dict."""
    for path_name in ["anomaly_kmeans.pkl", "anomaly_scaler.pkl", "anomaly_threshold.pkl"]:
        if not (MODELS_DIR / path_name).exists():
            raise FileNotFoundError(f"{path_name} not found. Run scripts/train_anomaly_model.py first.")

    txn_hash  = hash_identity(
        str(parsed.get("account_number", "unknown")),
        str(parsed.get("account_holder", ""))
    )

    features  = np.array([[parsed[c] for c in ANOMALY_FEATURES]], dtype=float)
    kmeans    = joblib.load(MODELS_DIR / "anomaly_kmeans.pkl")
    scaler    = joblib.load(MODELS_DIR / "anomaly_scaler.pkl")
    threshold = joblib.load(MODELS_DIR / "anomaly_threshold.pkl")

    X_scaled  = scaler.transform(features)
    dist      = float(np.min(kmeans.transform(X_scaled), axis=1)[0])
    is_anomaly = dist > threshold

    if is_anomaly:
        verdict        = "ANOMALY DETECTED"
        recommendation = "Transaction should be blocked immediately. Trigger OTP verification and alert the account holder."
    else:
        verdict        = "NORMAL TRANSACTION"
        recommendation = "Transaction falls within normal behavioral patterns. No action required."

    return {
        "action": "anomaly_detection",
        "txn_hash": txn_hash,
        "centroid_distance": round(dist, 4),
        "threshold": round(float(threshold), 4),
        "is_anomaly": is_anomaly,
        "verdict": verdict,
        "recommendation": recommendation,
        "features": dict(zip(ANOMALY_FEATURES, features[0].tolist()))
    }
