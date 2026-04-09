"""Supplemental gap-fillers that should never override official Board/Fed rows."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from core.ingest import SPEAKER_COLUMNS

BIS_YEARLY_ZIP = "https://www.bis.org/speeches/speeches_{year}.zip"
_WS = re.compile(r"\s+")
_ROLE_PREFIX_RE = re.compile(
    r"^(chair|chairman|vice chair|vice chairman|vice chair for supervision|vice chairman for supervision|governor|member)\s+",
    re.I,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def _norm_name(name: str) -> str:
    cleaned = _ROLE_PREFIX_RE.sub("", (name or "").strip())
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned.lower())
    return _WS.sub(" ", cleaned).strip()


def _fed_board_mask(df: pd.DataFrame) -> pd.Series:
    joined = (
        df["description"].fillna("").astype(str)
        + " "
        + df["text"].fillna("").astype(str)
        + " "
        + df["title"].fillna("").astype(str)
    )
    return joined.str.contains(
        r"Board of Governors of the Federal Reserve System|Federal Open Market Committee",
        case=False,
        regex=True,
        na=False,
    )


def load_bis_fed_rows(*, min_year: int, max_year: int, cache_dir: Path | None = None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for year in range(min_year, max_year + 1):
        url = BIS_YEARLY_ZIP.format(year=year)
        cache_path = cache_dir / f"bis_speeches_{year}.zip" if cache_dir else None
        blob: bytes
        if cache_path and cache_path.exists():
            blob = cache_path.read_bytes()
        else:
            r = requests.get(url, timeout=180)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            blob = r.content
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(blob)
        zf = zipfile.ZipFile(io.BytesIO(blob))
        csv_name = next((name for name in zf.namelist() if name.endswith(".csv")), "")
        if not csv_name:
            continue
        df = pd.read_csv(zf.open(csv_name))
        if df.empty:
            continue
        df = df[_fed_board_mask(df)].copy()
        if df.empty:
            continue
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["date", "domain", "participant", "document", "word_count"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y%m%d")
    out = out.dropna(subset=["date", "author", "text"])
    out["domain"] = "www.bis.org"
    out["participant"] = out["author"].astype(str).str.strip()
    out["document"] = (
        out["title"].fillna("").astype(str).str.strip() + ". " + out["text"].fillna("").astype(str).str.strip()
    ).str.strip()
    out["word_count"] = out["document"].astype(str).str.split().str.len().astype(int)
    out = out[out["word_count"] >= 80].copy()
    return out[["date", "domain", "participant", "document", "word_count"]]


def add_bis_gap_fillers(
    speaker_df: pd.DataFrame,
    *,
    min_year: int,
    max_year: int,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    bis = load_bis_fed_rows(min_year=min_year, max_year=max_year, cache_dir=cache_dir)
    if bis.empty:
        return pd.DataFrame(columns=SPEAKER_COLUMNS)

    # Keep BIS as a conservative post-2020 gap-filler only.
    bis = bis[bis["date"].astype(str) >= "20200101"].copy()
    if bis.empty:
        return pd.DataFrame(columns=SPEAKER_COLUMNS)

    official = speaker_df[speaker_df["domain"].astype(str) == "www.federalreserve.gov"][["date", "participant"]].copy()
    official["_name"] = official["participant"].astype(str).map(_norm_name)
    known_names = set(official["_name"])
    bis["_name"] = bis["participant"].astype(str).map(_norm_name)
    bis = bis[bis["_name"].isin(known_names)].copy()
    if bis.empty:
        return pd.DataFrame(columns=SPEAKER_COLUMNS)

    official["_d"] = pd.to_datetime(official["date"].astype(str), format="%Y%m%d", errors="coerce")
    bis["_d"] = pd.to_datetime(bis["date"].astype(str), format="%Y%m%d", errors="coerce")
    keep_rows: list[int] = []
    for idx, row in bis.iterrows():
        same_person = official[official["_name"] == row["_name"]]
        if same_person.empty:
            keep_rows.append(idx)
            continue
        nearby = (same_person["_d"] - row["_d"]).abs().dt.days <= 3
        if not nearby.any():
            keep_rows.append(idx)
    bis = bis.loc[keep_rows].copy()
    if bis.empty:
        return pd.DataFrame(columns=SPEAKER_COLUMNS)

    bis = bis.drop(columns=["_name", "_d"], errors="ignore")
    bis["fomc-ref-date"] = ""
    bis["decision"] = ""
    bis["high"] = ""
    bis["low"] = ""
    return bis[SPEAKER_COLUMNS]
