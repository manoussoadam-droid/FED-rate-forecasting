"""Financial sentiment: Loughran–McDonald file if provided; else keyword fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.config import LM_DICT_PATH
from core.text_clean import FED_KEYWORDS, count_keyword_hits

# Minimal fallback: coarse positive / negative / uncertain word lists (not LM)
_POS_FALLBACK = frozenset(
    "growth strong favorable improvement confidence expansion gains momentum resilient".split()
)
_NEG_FALLBACK = frozenset(
    "weak contraction decline risk recession stress downside uncertainty deterioration adverse".split()
)


def _load_lm_lexicon(path: str) -> dict[str, dict[str, float]] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        df = pd.read_csv(p, sep="\t", dtype=str, low_memory=False)
    except Exception:
        df = pd.read_csv(p, dtype=str, low_memory=False)
    if "Word" not in df.columns:
        return None
    df["Word"] = df["Word"].str.lower()
    lex: dict[str, dict[str, float]] = {}
    for col in ("Positive", "Negative", "Uncertainty", "Litigious", "Strong_Modal", "Weak_Modal"):
        if col not in df.columns:
            continue
        sub = df[df[col].astype(str).str.strip().isin(("1", "1.0", "TRUE", "True"))]
        for w in sub["Word"]:
            lex.setdefault(w, {})[col.lower()] = 1.0
    return lex if lex else None


_LM_CACHE: dict[str, dict[str, dict[str, float]] | None] = {}


def get_lm_lexicon() -> dict[str, dict[str, float]] | None:
    key = LM_DICT_PATH or ""
    if key not in _LM_CACHE:
        _LM_CACHE[key] = _load_lm_lexicon(key) if key else None
    return _LM_CACHE[key]


def score_lm_style(text: str) -> dict[str, Any]:
    """
    Returns scores in [-1, 1] style summary + counts.
    If LM file missing, uses fallback keyword buckets.
    """
    tokens = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split()
    lex = get_lm_lexicon()

    if lex:
        pos = neg = unc = 0
        for tok in tokens:
            entry = lex.get(tok)
            if not entry:
                continue
            if entry.get("positive"):
                pos += 1
            if entry.get("negative"):
                neg += 1
            if entry.get("uncertainty"):
                unc += 1
        total_fin = pos + neg + unc + 1e-9
        score = (pos - neg) / total_fin
        return {
            "method": "loughran_mcdonald_file",
            "positive_hits": pos,
            "negative_hits": neg,
            "uncertainty_hits": unc,
            "financial_sentiment_score": round(float(score), 4),
        }

    pos = sum(1 for t in tokens if t in _POS_FALLBACK)
    neg = sum(1 for t in tokens if t in _NEG_FALLBACK)
    hits = count_keyword_hits(text)
    total = pos + neg + 1e-9
    score = (pos - neg) / total
    return {
        "method": "fallback_keywords",
        "positive_hits": pos,
        "negative_hits": neg,
        "macro_keyword_hits": hits,
        "financial_sentiment_score": round(float(score), 4),
    }
