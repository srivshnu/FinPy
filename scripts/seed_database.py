"""
seed_database.py
Run ONCE to create the 3-table schema and populate it with
2000 realistic synthetic banking records.

Usage:
    python scripts/seed_database.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from logic.db_core import get_connection, init_schema

np.random.seed(42)
N = 2000

def seed():
    init_schema()
    conn = get_connection()
    cur = conn.cursor()

    # Check if already seeded
    cur.execute("SELECT COUNT(*) FROM customers")
    if cur.fetchone()[0] >= N:
        print("Database already seeded. Skipping.")
        cur.close()
        conn.close()
        return

    print("Seeding customers...")
    phone_set = set()
    customers = []
    while len(customers) < N:
        phone = f"9{np.random.randint(100000000, 999999999)}"
        if phone in phone_set:
            continue
        phone_set.add(phone)
        customers.append((
            f"Customer_{len(customers)}",
            phone,
            int(np.random.randint(21, 65)),
            round(float(np.random.uniform(120000, 2000000)), 2),
            int(np.random.randint(300, 900)),
            int(np.random.randint(0, 35)),
        ))

    cur.executemany("""
        INSERT INTO customers (full_name, phone_number, age, annual_income, credit_score, employment_years)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, customers)
    conn.commit()

    # Fetch inserted customer IDs
    cur.execute("SELECT customer_id, annual_income, credit_score, employment_years FROM customers")
    rows = cur.fetchall()

    print("Seeding loans...")
    loans = []
    for cid, income, cscore, emp in rows:
        debt    = round(float(np.random.uniform(0, income * 0.6)), 2)
        loan    = round(float(np.random.uniform(10000, income * 0.8)), 2)
        tenure  = int(np.random.choice([12, 24, 36, 48, 60, 84, 120, 180]))
        rate    = round(float(np.random.uniform(7.5, 24.0)), 2)
        # Logically sound default label
        risk = (debt / income) * 3 + (900 - cscore) / 150 - emp / 20
        default = int(risk + np.random.normal(0, 0.5) > 3.5)
        loans.append((cid, debt, loan, tenure, rate, default))

    cur.executemany("""
        INSERT INTO loans (customer_id, total_existing_debt, loan_amount_requested,
                           loan_tenure_months, interest_rate, default_status)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, loans)
    conn.commit()

    print("Seeding transactions...")
    rng = np.random.default_rng(99)
    txns = []
    for cid, income, _, _ in rows:
        n_txns = int(np.random.randint(1, 4))
        avg30  = round(float(np.random.uniform(200, income / 12 * 0.3)), 2)
        for _ in range(n_txns):
            acc = str(rng.integers(100000000000, 999999999999))
            is_fraud = np.random.rand() < 0.05  # 5% fraud
            if is_fraud:
                amount   = round(float(np.random.uniform(50000, 500000)), 2)
                hour     = int(np.random.randint(0, 5))
                dist     = round(float(np.random.uniform(2000, 10000)), 2)
                vel      = int(np.random.randint(30, 100))
            else:
                amount   = round(float(np.random.uniform(50, min(5000, avg30 * 3))), 2)
                hour     = int(np.random.randint(7, 22))
                dist     = round(float(np.random.uniform(0, 50)), 2)
                vel      = int(np.random.randint(1, 8))
            txns.append((acc, cid, amount, hour, dist, vel, avg30))

    cur.executemany("""
        INSERT INTO transactions (account_number, customer_id, transaction_amount,
                                  hour_of_day, distance_from_home_km,
                                  transactions_last_24h, avg_txn_amount_30d)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, txns)
    conn.commit()
    cur.close()
    conn.close()
    print(f"Done. Seeded {N} customers, {N} loans, {len(txns)} transactions.")

if __name__ == "__main__":
    seed()
