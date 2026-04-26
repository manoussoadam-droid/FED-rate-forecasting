"""Paths and flags from environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root = parent of core/
ROOT = Path(__file__).resolve().parent.parent


def _resolve_data_dir() -> Path:
    raw = os.environ.get("DATA_DIR", "").strip()
    if not raw:
        return (ROOT / "data").resolve()
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (ROOT / p).resolve()


DATA_DIR = _resolve_data_dir()

# Canonical columnar store (Parquet, partitioned by year) and audit DB.
PARQUET_DIR = DATA_DIR / "parquet"
FOMC_PARQUET_DIR = PARQUET_DIR / "fomc"
SPEAKER_PARQUET_DIR = PARQUET_DIR / "speaker"
AUDIT_DB = DATA_DIR / "audit.sqlite"
ERRORS_CSV = DATA_DIR / "errors.csv"

ARTIFACTS_DIR = ROOT / "artifacts"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
MODEL_META_PATH = ARTIFACTS_DIR / "model_meta.joblib"

LM_DICT_PATH = os.environ.get("LM_DICT_PATH", "").strip() or None
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip() or None

USE_XGBOOST = os.environ.get("USE_XGBOOST", "0").lower() in ("1", "true", "yes")

# Analysis limits (avoid huge payloads)
MAX_TEXT_CHARS_FOR_VECTOR = 100_000
MAX_WORDCLOUD_WORDS = 200

# --- External data API keys (all optional; graceful degradation if absent) ---
FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip() or None
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "").strip() or None
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "").strip() or None

# Flask API key — if set, all /api/v1/* routes require X-API-Key: <value>.
# Leave unset (or empty) during local development to skip enforcement.
API_KEY = os.environ.get("API_KEY", "").strip() or None

# FRED series fetched by the scheduler's daily job
FRED_DEFAULT_SERIES = [
    "DFEDTARL",   # Fed funds target range lower bound
    "DFEDTARU",   # Fed funds target range upper bound
    "FEDFUNDS",   # Effective federal funds rate
    "T10Y2Y",     # 10-year minus 2-year Treasury spread
    "UNRATE",     # Unemployment rate
    "CPIAUCSL",   # CPI all urban consumers
]
