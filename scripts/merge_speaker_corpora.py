#!/usr/bin/env python3
"""Merge Fed-extended speaker rows with non-Fed rows from a legacy full speaker pickle.

Strategy A extension (`extend_speaker_federalreserve.py`) is Fed-only. This script
re-attaches Bloomberg / press / etc. rows from the original-style corpus so
`speaker_doc.pkl` stays comparable to the historical multi-source dataset.

Example:

  python scripts/merge_speaker_corpora.py \\
    --fed data/speaker_doc_fed_extended.pkl \\
    --legacy data/speaker_doc_legacy.pkl \\
    --out data/speaker_doc.pkl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ingest import SPEAKER_COLUMNS  # noqa: E402


def fed_mask(df: pd.DataFrame) -> pd.Series:
    d = df["domain"].astype(str).str.lower()
    return (
        d.str.contains("federalreserve", na=False)
        | d.str.endswith("fed.org", na=False)
        | d.str.contains(".fed.org", na=False)
        | d.str.contains("frb.org", na=False)
        | d.str.contains("frbsf.org", na=False)
        | d.str.contains("frbatlanta.org", na=False)
    )


def ensure_yyyymmdd(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if hasattr(val, "strftime"):
        return val.strftime("%Y%m%d")  # type: ignore[union-attr]
    s = str(val).strip().replace(".0", "")
    if re.fullmatch(r"\d{8}", s):
        return s
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return s
    return pd.Timestamp(ts).strftime("%Y%m%d")


def coerce_speaker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["date"].map(ensure_yyyymmdd)
    out["fomc-ref-date"] = out["fomc-ref-date"].map(ensure_yyyymmdd)
    for c in ("decision", "high", "low", "domain", "participant", "document"):
        out[c] = out[c].astype(str)
    out["word_count"] = out["word_count"].astype("int64")
    return out.sort_values(["date", "domain", "participant"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fed", type=Path, required=True, help="Fed-extended pickle (e.g. from extend_speaker_federalreserve.py)")
    p.add_argument("--legacy", type=Path, required=True, help="Full legacy speaker pickle (multi-domain)")
    p.add_argument("--out", type=Path, required=True, help="Output pickle path")
    args = p.parse_args()

    sp_ext = pd.read_pickle(args.fed)
    sp_old = pd.read_pickle(args.legacy)
    nonfed = sp_old[~fed_mask(sp_old)].copy()
    merged = pd.concat([sp_ext, nonfed], axis=0, ignore_index=True)
    merged = coerce_speaker(merged)

    missing = [c for c in SPEAKER_COLUMNS if c not in merged.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    if not merged["date"].astype(str).str.fullmatch(r"\d{8}").all():
        raise SystemExit("Invalid date values")
    if not merged["fomc-ref-date"].astype(str).str.fullmatch(r"\d{8}").all():
        raise SystemExit("Invalid fomc-ref-date values")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_pickle(args.out)
    print(f"Wrote {args.out} rows={len(merged)} (fed_ext={len(sp_ext)} + non_fed={len(nonfed)})")


if __name__ == "__main__":
    main()
