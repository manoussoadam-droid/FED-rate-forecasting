"""Load FOMC/speaker pickles, tidy dates, train/test split."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.utils import shuffle

from core.config import FOMC_PICKLE, SPEAKER_PICKLE

# `type` (statement / minutes / …) is optional in pickles; ML only uses decision + document + word_count.
FOMC_COLUMNS = ["date", "decision", "high", "low", "document", "word_count"]
FOMC_COLUMNS_LEGACY = ["date", "type", "decision", "high", "low", "document", "word_count"]
SPEAKER_COLUMNS = [
    "fomc-ref-date",
    "date",
    "decision",
    "high",
    "low",
    "domain",
    "participant",
    "document",
    "word_count",
]


def _validate_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}. Found: {list(df.columns)}")


def load_fomc(path: str | None = None) -> pd.DataFrame:
    p = path or str(FOMC_PICKLE)
    df = pd.read_pickle(p)
    has_type = "type" in df.columns
    expected = FOMC_COLUMNS_LEGACY if has_type else FOMC_COLUMNS
    _validate_columns(df, expected, "fomc_doc.pkl")
    return df


def load_speaker(path: str | None = None) -> pd.DataFrame:
    p = path or str(SPEAKER_PICKLE)
    df = pd.read_pickle(p)
    _validate_columns(df, SPEAKER_COLUMNS, "speaker_doc.pkl")
    return df


def parse_yyyymmdd(series: pd.Series) -> pd.Series:
    """Coerce YYYYMMDD strings to datetime; invalid -> NaT."""

    s = series.astype(str).str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def add_parsed_dates(fomc: pd.DataFrame, speaker: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = fomc.copy()
    sp = speaker.copy()
    f["date_parsed"] = parse_yyyymmdd(f["date"])
    sp["date_parsed"] = parse_yyyymmdd(sp["date"])
    sp["fomc_ref_parsed"] = parse_yyyymmdd(sp["fomc-ref-date"])
    return f, sp


def build_combined_eda_frame(fomc: pd.DataFrame, speaker: pd.DataFrame) -> pd.DataFrame:
    """Concat FOMC + speaker for EDA; column ``type`` is FOMC doc kind or speaker ``domain``."""
    f_part = fomc[["date", "document", "word_count", "decision"]].copy()
    if "type" in fomc.columns:
        f_part["type"] = fomc["type"].astype(str)
    else:
        f_part["type"] = "fomc"
    s_part = speaker[["date", "domain", "document", "word_count", "decision"]].copy()
    s_part = s_part.rename(columns={"domain": "type"})
    return pd.concat([f_part, s_part], axis=0)


def build_train_test_split(
    fomc: pd.DataFrame,
    speaker: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Notebook logic:
    - train = all FOMC rows + first 70% of speaker (by row order)
    - test = remaining 30% speaker
    Columns: decision, word_count, document
    """
    train_speaker_len = int(len(speaker) * 0.7)
    train_df = pd.concat(
        [
            fomc[["decision", "word_count", "document"]],
            speaker[["decision", "word_count", "document"]].iloc[:train_speaker_len],
        ],
        axis=0,
    )
    test_df = speaker[["decision", "word_count", "document"]].iloc[train_speaker_len:]
    train_df = shuffle(train_df, random_state=random_state)
    test_df = shuffle(test_df, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def random_corpus_row(fomc: pd.DataFrame, speaker: pd.DataFrame) -> dict[str, Any]:
    """Single random document with metadata for /sample."""
    combined = build_combined_eda_frame(fomc, speaker)
    row = combined.sample(1).iloc[0]
    return {
        "date": str(row["date"]),
        "type": str(row["type"]),
        "decision": str(row["decision"]),
        "word_count": int(row["word_count"]),
        "document": str(row["document"])[:8000],
    }
