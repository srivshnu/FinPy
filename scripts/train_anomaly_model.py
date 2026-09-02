"""
train_anomaly_model.py
Run ONCE after seed_database.py to produce:
    models/anomaly_kmeans.pkl
    models/anomaly_scaler.pkl
    models/anomaly_threshold.pkl

Usage:
    python scripts/train_anomaly_model.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from logic.db_core import query_to_df, init_schema
from logic.hasher import anonymise_dataframe

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "transaction_amount", "hour_of_day", "distance_from_home_km",
    "transactions_last_24h", "avg_txn_amount_30d"
]

init_schema()
df = query_to_df("""
    SELECT
        t.account_number,
        c.full_name,
        t.transaction_amount,
        t.hour_of_day,
        t.distance_from_home_km,
        t.transactions_last_24h,
        t.avg_txn_amount_30d
    FROM transactions t
    JOIN customers c ON t.customer_id = c.customer_id
""")

print(f"Loaded {len(df)} transactions from DB.")

# Anonymise PII
df = anonymise_dataframe(df, id_col="account_number", pii_cols=["full_name"])
print("PII anonymised.")

X = df[FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

distances  = np.min(kmeans.transform(X_scaled), axis=1)
threshold  = float(np.percentile(distances, 95))
print(f"Anomaly distance threshold (95th pct): {threshold:.4f}")

joblib.dump(kmeans,    MODELS_DIR / "anomaly_kmeans.pkl")
joblib.dump(scaler,    MODELS_DIR / "anomaly_scaler.pkl")
joblib.dump(threshold, MODELS_DIR / "anomaly_threshold.pkl")
print(f"Saved -> {MODELS_DIR}")
