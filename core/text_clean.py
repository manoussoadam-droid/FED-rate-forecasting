"""Regex-based cleaning and extraction for ML / analysis."""

from __future__ import annotations

import re
from typing import Iterable

# Fed / macro keywords for simple fallback sentiment / tagging
FED_KEYWORDS = (
    "inflation",
    "employment",
    "labor market",
    "policy rate",
    "federal funds",
    "accommodation",
    "disinflation",
    "recession",
    "growth",
    "outlook",
    "uncertainty",
)

_WS = re.compile(r"\s+")
_NON_ALNUM_SPACE = re.compile(r"[^\w\s]", re.UNICODE)
_DATE_MDY = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
_DATE_ISOISH = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
_PCT = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_NUMBER_BP = re.compile(r"\b\d+\s*basis\s+points?\b", re.I)


def collapse_whitespace(text: str) -> str:
    return _WS.sub(" ", text.strip())


def strip_boilerplate_noise(text: str) -> str:
    """Remove repeated non-alphanumeric clutter; keep words and spaces."""
    t = _NON_ALNUM_SPACE.sub(" ", text)
    return collapse_whitespace(t)


def extract_dates(text: str) -> list[str]:
    out = [m.group(0) for m in _DATE_MDY.finditer(text)]
    out += [m.group(0) for m in _DATE_ISOISH.finditer(text)]
    return out


def extract_percentages(text: str) -> list[str]:
    return [m.group(0) for m in _PCT.finditer(text)]


def extract_basis_points(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER_BP.finditer(text)]


def count_keyword_hits(text: str, keywords: Iterable[str] = FED_KEYWORDS) -> dict[str, int]:
    lower = text.lower()
    return {kw: len(re.findall(re.escape(kw.lower()), lower)) for kw in keywords}


def clean_for_ml(text: str, max_chars: int | None = None) -> str:
    """Light normalization before vectorization."""
    t = strip_boilerplate_noise(text)
    if max_chars is not None:
        t = t[:max_chars]
    return t
