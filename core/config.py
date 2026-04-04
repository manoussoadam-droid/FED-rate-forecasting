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
FOMC_PICKLE = DATA_DIR / "fomc_doc.pkl"
SPEAKER_PICKLE = DATA_DIR / "speaker_doc.pkl"

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
