"""Load FOMC/speaker corpora from the canonical Parquet store.

The canonical store is Parquet (partitioned by year, see :mod:`core.repository`).
Legacy pickle files are no longer used as a fallback; if the Parquet store is
absent the functions raise :class:`FileNotFoundError` with rebuild instructions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.utils import shuffle

from core.config import FOMC_PARQUET_DIR, SPEAKER_PARQUET_DIR

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


def _parquet_dir_has_data(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def _fomc_from_parquet() -> pd.DataFrame:
    # Import lazily so `core.ingest` stays usable even if pyarrow is missing.
    from core.repository import DocumentRepository

    repo = DocumentRepository()
    df = repo.read_fomc()
    if df.empty:
        return pd.DataFrame(columns=FOMC_COLUMNS_LEGACY)
    out = pd.DataFrame(
        {
            "date": df["date"].astype(str),
            "type": df["document_type"].astype(str),
            "decision": df["decision"].astype(str),
            "high": df["high"].astype(str),
            "low": df["low"].astype(str),
            "document": df["text_content"].astype(str),
            "word_count": df["text_content"].astype(str).map(lambda t: len(t.split())).astype("int64"),
        }
    )
    return out


def _speaker_from_parquet() -> pd.DataFrame:
    from core.repository import DocumentRepository

    repo = DocumentRepository()
    df = repo.read_speaker()
    if df.empty:
        return pd.DataFrame(columns=SPEAKER_COLUMNS)
    out = pd.DataFrame(
        {
            "fomc-ref-date": df["fomc_ref_date"].astype(str),
            "date": df["date"].astype(str),
            "decision": df["decision"].astype(str),
            "high": df["high"].astype(str),
            "low": df["low"].astype(str),
            "domain": df["domain"].astype(str),
            "participant": df["participant"].astype(str),
            "document": df["text_content"].astype(str),
            "word_count": df["text_content"].astype(str).map(lambda t: len(t.split())).astype("int64"),
        }
    )
    return out


def load_fomc(path: str | None = None) -> pd.DataFrame:
    """Load FOMC corpus from Parquet.

    ``path`` is kept for test fixtures that supply a pre-built DataFrame
    serialised with ``pd.DataFrame.to_parquet``; pass a ``.parquet`` file
    path, not a pickle.  Production callers should leave ``path=None``.
    """
    if path is not None:
        df = pd.read_parquet(path)
        has_type = "type" in df.columns
        expected = FOMC_COLUMNS_LEGACY if has_type else FOMC_COLUMNS
        _validate_columns(df, expected, str(path))
        return df
    if not _parquet_dir_has_data(FOMC_PARQUET_DIR):
        raise FileNotFoundError(
            f"FOMC Parquet store not found at {FOMC_PARQUET_DIR}. "
            "Run `python scripts/rebuild_database.py` to populate it."
        )
    df = _fomc_from_parquet()
    if df.empty:
        raise FileNotFoundError(
            f"FOMC Parquet store at {FOMC_PARQUET_DIR} exists but returned no rows. "
            "Run `python scripts/rebuild_database.py` to rebuild."
        )
    _validate_columns(df, FOMC_COLUMNS_LEGACY, "fomc parquet")
    return df


def load_speaker(path: str | None = None) -> pd.DataFrame:
    """Load speaker corpus from Parquet.

    ``path`` is kept for test fixtures; see :func:`load_fomc` for details.
    """
    if path is not None:
        df = pd.read_parquet(path)
        _validate_columns(df, SPEAKER_COLUMNS, str(path))
        return df
    if not _parquet_dir_has_data(SPEAKER_PARQUET_DIR):
        raise FileNotFoundError(
            f"Speaker Parquet store not found at {SPEAKER_PARQUET_DIR}. "
            "Run `python scripts/rebuild_database.py` to populate it."
        )
    df = _speaker_from_parquet()
    if df.empty:
        raise FileNotFoundError(
            f"Speaker Parquet store at {SPEAKER_PARQUET_DIR} exists but returned no rows. "
            "Run `python scripts/rebuild_database.py` to rebuild."
        )
    _validate_columns(df, SPEAKER_COLUMNS, "speaker parquet")
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
