"""
train_credit_model.py
Run ONCE after seed_database.py to produce models/credit_model.pkl

Usage:
    python scripts/train_credit_model.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from logic.db_core import query_to_df, init_schema
from logic.hasher import anonymise_dataframe

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Pull training data from the relational DB using a JOIN
init_schema()
df = query_to_df("""
    SELECT
        c.phone_number,
        c.full_name,
        c.age,
        c.annual_income,
        c.credit_score,
        c.employment_years,
        l.total_existing_debt,
        l.loan_amount_requested,
        l.loan_tenure_months,
        l.interest_rate,
        l.default_status
    FROM loans l
    JOIN customers c ON l.customer_id = c.customer_id
""")

print(f"Loaded {len(df)} rows from DB. Default rate: {df['default_status'].mean():.2%}")

# Anonymise PII before training
df = anonymise_dataframe(df, id_col="phone_number", pii_cols=["full_name"])
print("PII anonymised.")

FEATURES = [
    "age", "annual_income", "credit_score", "employment_years",
    "total_existing_debt", "loan_amount_requested",
    "loan_tenure_months", "interest_rate"
]
X = df[FEATURES]
y = df["default_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n-- Classification Report --")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")

joblib.dump(model, MODELS_DIR / "credit_model.pkl")
print(f"\nSaved -> {MODELS_DIR / 'credit_model.pkl'}")
