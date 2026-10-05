<div align="center">
  <h1>FinPy</h1>
  <p><b>NLP-Assisted Risk & Anomaly Detection Web Application</b></p>

  ![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
  ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat&logo=mysql&logoColor=white)
  ![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-green)
  ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
  ![Gemini](https://img.shields.io/badge/Google_GenAI-Gemini-blue?logo=google)
</div>

---

## Overview

FinPy is a single-page banking intelligence app. A user types a plain-English scenario (a loan applicant, a transaction, or a data question), picks an action, and clicks **Run**. FinPy then:

1. Uses **Google Gemini** to turn the free text into a strict, typed JSON record.
2. **Anonymises PII** with HMAC-SHA256 so raw identifiers never reach the model layer.
3. Routes the record to a **pre-trained ML model** (XGBoost or K-Means) or to a **live MySQL database** and shows the result as metrics, charts and tables.

> The data is **synthetic** (generated with Faker). FinPy is a portfolio/learning project, not a production credit decisioning system.

---

## Features

| Action | What it does | Tech |
|---|---|---|
| **Credit Risk** | Estimates default probability and gives a `LOW` / `MODERATE` / `HIGH` verdict with an underwriting recommendation | XGBoost |
| **Anomaly / Fraud** | Flags transactions that sit too far from normal behaviour clusters | K-Means + StandardScaler |
| **Text-to-SQL** | Converts a plain-English question into MySQL, runs it, shows a DataFrame, heatmap and CSV/Excel export | Gemini + MySQL + Pandas |

### Privacy by design
- Names, phone numbers and account numbers are never sent to the models.
- `HMAC-SHA256(pepper, identifier)` derives a deterministic 8-character PIN; a SHA-256 session hash is built from the PIN, a timestamp and the other PII fields.
- The pepper (`HASH_PEPPER`) lives only in `.env`, which is git-ignored.

### Model details
- **Credit model:** XGBoost with `scale_pos_weight` for class imbalance. Reported on the synthetic dataset: **0.98 ROC-AUC**, **0.93 F1**. Features: `age`, `annual_income`, `credit_score`, `employment_years`, `total_existing_debt`, `loan_amount_requested`, `loan_tenure_months`, `interest_rate`.
- **Risk bands:** default probability `>= 70%` is HIGH, `>= 40%` is MODERATE, otherwise LOW.
- **Anomaly model:** K-Means on `transaction_amount`, `hour_of_day`, `distance_from_home_km`, `transactions_last_24h`, `avg_txn_amount_30d`. A transaction is anomalous if its distance to the nearest centroid exceeds the **95th-percentile** training threshold.

---

## Architecture

```text
 User text
    |
    v
 [ Streamlit UI  ui/copilot.py ]
    |
    v
 [ Gemini parser  logic/inference_logic.py ] --> typed JSON features
    |
    v
 [ HMAC-SHA256 hasher  logic/hasher.py ] --> anonymised key
    |
    +--> XGBoost credit model ----> verdict + probability
    +--> K-Means anomaly model ---> verdict + centroid distance
    +--> Text-to-SQL (eda_logic) -> MySQL -> DataFrame / charts
```

The UI layer contains no business logic: everything in `logic/` is plain Python with no Streamlit imports.

### Database schema (MySQL, 3 tables)

```text
 +----------------------+         +------------------------------+
 |      customers       |         |            loans             |
 +----------------------+         +------------------------------+
 | customer_id (PK)     |<--+     | loan_id (PK)                 |
 | full_name (PII)      |   |     | customer_id (FK) ------------+
 | phone_number (PII)   |   |     | total_existing_debt          |
 | age                  |   |     | loan_amount_requested        |
 | annual_income        |   |     | loan_tenure_months           |
 | credit_score         |   |     | interest_rate                |
 | employment_years     |   |     | default_status               |
 +----------------------+   |     +------------------------------+
                            |
                            |     +------------------------------+
                            |     |         transactions         |
                            |     +------------------------------+
                            |     | transaction_id (PK)          |
                            +---- | customer_id (FK)             |
                                  | account_number (PII)         |
                                  | transaction_amount           |
                                  | hour_of_day                  |
                                  | distance_from_home_km        |
                                  | transactions_last_24h        |
                                  | avg_txn_amount_30d           |
                                  +------------------------------+
```

---

## Project Structure

```text
FinPy/
|-- app.py                      # Streamlit entry point (router)
|-- requirements.txt
|-- .env.example                # Template for credentials
|-- logic/                      # Business + AI logic (no UI code)
|   |-- db_core.py              # MySQL connection, auto-create DB + schema
|   |-- eda_logic.py            # Text-to-SQL, EDA, exporters
|   |-- hasher.py               # HMAC-SHA256 PII anonymisation
|   |-- inference_logic.py      # LLM parsing + model inference
|   |-- llm_client.py           # Gemini client
|   |-- plot_logic.py           # Matplotlib charts
|-- ui/
|   |-- copilot.py              # Single-page UI
|-- scripts/
|   |-- seed_database.py        # Seeds 2,000 synthetic customers/loans/transactions
|   |-- train_credit_model.py   # Trains XGBoost -> models/credit_model.pkl
|   |-- train_anomaly_model.py  # Trains K-Means -> models/anomaly_*.pkl
|-- models/                     # Generated .pkl files (git-ignored)
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- A running MySQL server
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 1. Clone and install
```powershell
git clone https://github.com/srivshnu/FinPy.git
cd FinPy
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`
Copy `.env.example` to `.env` and fill it in:
```env
GEMINI_API_KEY=your_gemini_api_key
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DB=finpy
HASH_PEPPER=any_long_random_secret
```
The `finpy` database and its tables are created automatically on first connection.

### 3. Seed data and train models (once)
```powershell
python scripts/seed_database.py          # skips if already seeded
python scripts/train_credit_model.py
python scripts/train_anomaly_model.py
```

### 4. Launch
```powershell
streamlit run app.py
```
Open `http://localhost:8501`.

---

## Example Prompts

- **Credit Risk:** `Rahul, 29, earns 8 lakh a year, credit score 710, 4 years employed, existing debt 1.5 lakh, wants a 5 lakh loan for 36 months at 11%.`
- **Anomaly:** `A transaction of 95,000 at 3 AM, 800 km from home, 9 transactions in the last 24 hours, usual average is 2,000.`
- **Text-to-SQL:** `Show the average credit score of customers whose loans defaulted.`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Access denied for user 'root'` | Set the correct `MYSQL_PASSWORD` in `.env` |
| `Fatal error in launcher` for streamlit | The `.venv` was moved; delete it and recreate (step 1) |
| `404 model not found` from Gemini | The model name in `logic/llm_client.py` was retired; pick a current one from `genai.list_models()` |
| `credit_model.pkl not found` | Run the training scripts (step 3) |

---

## Limitations

- All data is synthetic, so metrics (0.98 ROC-AUC) will not transfer to real portfolios.
- The LLM infers defaults for missing fields; unusual inputs can produce unrealistic features.
- Generated SQL is executed against the database; run it with a read-only MySQL user in any shared setting.
- `google-generativeai` is deprecated; migrating to `google-genai` is a planned improvement.

---

## Security Notes

- Secrets are loaded with `python-dotenv`; `.env`, `models/` and `data/` are git-ignored.
- Never commit your real `.env`. Rotate your Gemini key if it was ever shared.
