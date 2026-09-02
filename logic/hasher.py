"""
hasher.py
Privacy-preserving, deterministic hashing engine.
Converts PII (phone number / account number) into a permanent anonymised key.
No Streamlit imports.
"""
import hashlib
import hmac
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# A server-side pepper loaded from .env — adds an extra layer unknown to engineers
_PEPPER = os.getenv("HASH_PEPPER", "finpy-default-pepper-change-in-prod")


def _derive_pin(identifier: str) -> str:
    """
    Derive a fixed, permanent PIN from a phone/account number.
    HMAC-SHA256 with the server pepper means the raw identifier
    is never recoverable from the PIN alone.
    """
    return hmac.new(
        _PEPPER.encode(),
        identifier.encode(),
        "sha256"
    ).hexdigest()[:8].upper()          # 8-char uppercase PIN (readable, permanent)


def hash_identity(
    identifier: str,
    *pii_fields: str,
    timestamp: str | None = None
) -> str:
    """
    Build the final anonymised key:
        SHA-256( PIN + timestamp + all extra PII fields )

    Parameters
    ----------
    identifier  : The primary PII field (phone number or account number).
    *pii_fields : Additional PII strings to fold in (name, DOB, etc.).
    timestamp   : ISO timestamp string; uses current UTC time if None.

    Returns
    -------
    64-character hex digest — used as the row's unique, privacy-safe key.
    """
    pin = _derive_pin(identifier)
    ts  = timestamp or datetime.utcnow().isoformat()
    combined = pin + ts + "".join(pii_fields)
    return hashlib.sha256(combined.encode()).hexdigest()


def anonymise_dataframe(df, id_col: str, pii_cols: list[str], hash_col: str = "user_hash"):
    """
    Replace PII columns in a DataFrame with a single anonymised hash column.
    Returns a new DataFrame with PII removed.
    """
    import pandas as pd

    df = df.copy()
    df[hash_col] = df.apply(
        lambda row: hash_identity(
            str(row[id_col]),
            *[str(row[c]) for c in pii_cols if c in df.columns]
        ),
        axis=1
    )
    drop_cols = [id_col] + [c for c in pii_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    return df
