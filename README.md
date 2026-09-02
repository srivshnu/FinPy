# FinPy — NLP Assisted Risk & Anomaly Detection Web Application

A unified, AI-driven banking intelligence copilot built with **Python**, **Streamlit**, **MySQL**, **XGBoost**, **Scikit-Learn**, and the **Google Gemini API**. 

FinPy transforms natural language customer scenarios and transaction descriptions into structured feature vectors, enforces strict cryptographic PII privacy, and executes real-time credit risk estimation, fraud anomaly scoring, or natural language database querying.

---

## 🌟 Key Architecture & Features

### 1. Unified Agentic Copilot Chat Interface (`ui/copilot.py`)
- Single-page conversational banking workstation.
- Users input natural language descriptions of customer loan applications, transaction alerts, or database inquiries.
- Interactively routes queries to the appropriate machine learning pipeline or relational database engine.

### 2. Privacy-Preserving PII Anonymization Engine (`logic/hasher.py`)
- **Zero-PII Exposure:** Raw customer identifiers (phone numbers, account numbers, names) never reach the machine learning models or third-party APIs.
- Uses **HMAC-SHA256** with a server-side pepper key (`HASH_PEPPER`) to derive deterministic 8-character PINs and 64-character session hashes.
- Ensures irreversible, privacy-safe identity tracking while completely removing personal bias from risk scoring.

### 3. Natural Language Feature Extraction (`logic/inference_logic.py`)
- Leverages **Google Gemini API** (`gemini-1.5-flash`) via strict prompt engineering to parse unstructured, free-form text into typed JSON feature objects.
- Automatically infers domain-sound defaults for omitted parameters based on standard banking baselines.

### 4. Credit Risk Classification Pipeline (`scripts/train_credit_model.py`)
- Pre-trained **XGBoost Classifier** trained on relational data (`customers JOIN loans`).
- Features: `age`, `annual_income`, `credit_score`, `employment_years`, `total_existing_debt`, `loan_amount_requested`, `loan_tenure_months`, `interest_rate`.
- Handles class imbalance using `scale_pos_weight`, achieving **0.98 ROC-AUC** and 0.93 F1-score.
- Outputs default probability percentage, risk level verdict (`LOW`, `MODERATE`, `HIGH`), and actionable underwriting recommendations.

### 5. Transaction Anomaly & Fraud Detection (`scripts/train_anomaly_model.py`)
- Pre-trained **K-Means Clustering** model coupled with `StandardScaler` normalisation.
- Features: `transaction_amount`, `hour_of_day`, `distance_from_home_km`, `transactions_last_24h`, `avg_txn_amount_30d`.
- Evaluates Euclidean distance from normal behavior centroids against a 95th-percentile anomaly threshold.
- Visualises transaction positioning via inline Matplotlib scatter plots.

### 6. Text-to-SQL Analytics Engine (`logic/eda_logic.py`)
- Converts plain-English queries into valid MySQL queries using LLM schema context.
- Executes queries against live MySQL tables and presents interactive Pandas DataFrames.
- Offers automatic dataset correlation heatmaps and instant CSV/Excel data exports.

---

## 🗃 Relational Database Schema (`logic/db_core.py`)

FinPy utilizes a 3-table relational schema designed for Core Banking Systems (CBS):

```
 ┌──────────────────────┐         ┌──────────────────────────────┐
 │      customers       │         │            loans             │
 ├──────────────────────┤         ├──────────────────────────────┤
 │ customer_id (PK)     │◄──┐     │ loan_id (PK)                 │
 │ full_name (PII)      │   │     │ customer_id (FK) ────────────┘
 │ phone_number (PII)   │   │     │ total_existing_debt          │
 │ age                  │   │     │ loan_amount_requested        │
 │ annual_income        │   │     │ loan_tenure_months           │
 │ credit_score         │   │     │ interest_rate                │
 │ employment_years     │   │     │ default_status               │
 └──────────────────────┘   │     └──────────────────────────────┘
                            │
                            │     ┌──────────────────────────────┐
                            │     │         transactions         │
                            │     ├──────────────────────────────┤
                            │     │ transaction_id (PK)          │
                            └────┤ customer_id (FK)             │
                                  │ account_number (PII)         │
                                  │ transaction_amount           │
                                  │ hour_of_day                  │
                                  │ distance_from_home_km        │
                                  │ transactions_last_24h        │
                                  │ avg_txn_amount_30d           │
                                  └──────────────────────────────┘
```

---

## 📂 Project Directory Structure

```
FinPy/
├── app.py                          # Main Streamlit application router
├── .env.example                    # Template for environment credentials
├── .gitignore                      # Excludes .env, models/, and data/
├── requirements.txt                # Production dependencies
├── README.md                       # Documentation
├── logic/                          # Core business & AI logic (No UI code)
│   ├── db_core.py                  # MySQL schema management & connection
│   ├── eda_logic.py                # Text-to-SQL, EDA reporting, format exporters
│   ├── hasher.py                   # HMAC-SHA256 cryptographic PII anonymization
│   ├── inference_logic.py          # LLM text parser + model inference handlers
│   ├── llm_client.py               # Centralized Google Gemini API client
│   └── plot_logic.py               # Matplotlib chart generators
├── ui/                             # Streamlit UI layer
│   └── copilot.py                  # Single-page Agentic Copilot interface
├── scripts/                        # Offline setup & training scripts
│   ├── seed_database.py            # Generates 2,000 synthetic CBS records in MySQL
│   ├── train_credit_model.py       # Trains & saves models/credit_model.pkl
│   └── train_anomaly_model.py      # Trains & saves models/anomaly_*.pkl
├── models/                         # Serialized model binaries (.pkl)
└── data/                           # Local reference datasets
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
- Python 3.10+ installed.
- MySQL Server installed and running locally or remotely.

### 2. Environment Configuration
Clone the repository and set up environment variables:

```bash
git clone https://github.com/srivshnu/FinPy.git
cd FinPy
```

Create a `.env` file in the root directory (refer to `.env.example`):
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DB=finpy
HASH_PEPPER=your_secret_pepper_string_here
```

### 3. Installation
Create and activate a virtual environment, then install dependencies:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Database Seeding & Model Training (Run Once Offline)
Initialize the MySQL schema, seed synthetic records, and train the ML models:

```bash
# Step 1: Seed MySQL database (2,000 customers, loans, transactions)
python scripts/seed_database.py

# Step 2: Train XGBoost Credit Risk Model
python scripts/train_credit_model.py

# Step 3: Train K-Means Anomaly Detection Model
python scripts/train_anomaly_model.py
```

### 5. Running the Application
Launch the Streamlit Agentic Banking Copilot:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🛡 Security & Privacy Statement

- **API Keys & Credentials:** Strictly managed via `python-dotenv` and ignored in `.gitignore`.
- **PII Protection:** Identifiers are stripped and replaced with HMAC-SHA256 digests prior to external API transmission or model inference.
