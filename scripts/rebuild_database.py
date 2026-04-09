#!/usr/bin/env python3
"""Refresh all dataset files used by the project.

This is the one script to rerun when new data is available.

Outputs in `data/`:
- `fomc_doc.pkl` / `speaker_doc.pkl`: fullest working pair for the app/training
- `fomc_doc_original.pkl` / `speaker_doc_original.pkl`: original public FedNLP pair
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import DATA_DIR, FOMC_PICKLE, SPEAKER_PICKLE  # noqa: E402
from core.ingest import FOMC_COLUMNS_LEGACY, SPEAKER_COLUMNS, load_fomc, load_speaker  # noqa: E402
from rebuild.assemble import (  # noqa: E402
    build_fomc_dataframe,
    build_speaker_dataframe,
    merge_speakers_nearest_fomc,
    validate_frames,
)
from rebuild.fred_rates import load_target_range_frame  # noqa: E402
from rebuild.supplements import add_bis_gap_fillers  # noqa: E402
from scraping.fed_official import FOMC_CALENDAR_URL, FedHttpSession  # noqa: E402
from scraping.nonfed_accessible import fetch_article  # noqa: E402

PUBLIC_FOMC_URL = "https://raw.githubusercontent.com/usydnlp/FedNLP/main/resources/fomc_doc.pkl"
PUBLIC_SPEAKER_URL = "https://raw.githubusercontent.com/usydnlp/FedNLP/main/resources/speaker_doc.pkl"
ORIGINAL_FOMC_PATH = DATA_DIR / "fomc_doc_original.pkl"
ORIGINAL_SPEAKER_PATH = DATA_DIR / "speaker_doc_original.pkl"
SEED_TSV = ROOT / "scripts" / "nonfed_accessible_seed_urls.tsv"
BACKUP_DIR = ROOT / "artifacts" / "data_backups"
BACKUP_FOMC = BACKUP_DIR / "fomc_doc.last_backup.pkl"
BACKUP_SPEAKER = BACKUP_DIR / "speaker_doc.last_backup.pkl"


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


def backup_current_pair() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if FOMC_PICKLE.exists():
        shutil.copy2(FOMC_PICKLE, BACKUP_FOMC)
    if SPEAKER_PICKLE.exists():
        shutil.copy2(SPEAKER_PICKLE, BACKUP_SPEAKER)


def download_if_needed(url: str, path: Path, *, refresh: bool) -> None:
    if path.exists() and not refresh:
        return
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)


def _coerce_fomc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "type" not in out.columns:
        out["type"] = "statement"
    out["date"] = out["date"].map(ensure_yyyymmdd)
    for c in ("type", "decision", "high", "low", "document"):
        out[c] = out[c].astype(str)
    out["word_count"] = out["word_count"].astype("int64")
    return out[FOMC_COLUMNS_LEGACY]


def _coerce_speaker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["date"].map(ensure_yyyymmdd)
    out["fomc-ref-date"] = out["fomc-ref-date"].map(ensure_yyyymmdd)
    for c in ("decision", "high", "low", "domain", "participant", "document"):
        out[c] = out[c].astype(str)
    out["word_count"] = out["word_count"].astype("int64")
    return out[SPEAKER_COLUMNS]


def merge_public_fomc(current_fomc: pd.DataFrame, public_fomc: pd.DataFrame) -> pd.DataFrame:
    current_fomc = _coerce_fomc(current_fomc)
    public_fomc = _coerce_fomc(public_fomc)
    legacy_non_statement = public_fomc[public_fomc["type"].astype(str) != "statement"].copy()
    merged = pd.concat([current_fomc, legacy_non_statement], axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "type"], keep="first")
    return merged.sort_values(["date", "type"], kind="mergesort").reset_index(drop=True)


def merge_public_speaker(current_speaker: pd.DataFrame, public_speaker: pd.DataFrame) -> pd.DataFrame:
    current_speaker = _coerce_speaker(current_speaker)
    public_speaker = _coerce_speaker(public_speaker)
    nonfed = public_speaker[~fed_mask(public_speaker)].copy()
    merged = pd.concat([current_speaker, nonfed], axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "domain", "participant", "document"], keep="first")
    return merged.sort_values(["date", "domain", "participant"], kind="mergesort").reset_index(drop=True)


def nearest_fomc_label(date_ymd: str, fomc_df: pd.DataFrame) -> tuple[str, str, str, str]:
    base = fomc_df[["date", "decision", "high", "low"]].copy()
    base["date"] = base["date"].astype(str)
    base["_d"] = pd.to_datetime(base["date"], format="%Y%m%d", errors="coerce")
    base = base.dropna(subset=["_d"]).sort_values("_d").reset_index(drop=True)
    target = pd.Timestamp(date_ymd)
    diffs = (base["_d"] - target).abs()
    row = base.loc[int(diffs.idxmin())]
    return str(row["date"]), str(row["decision"]), str(row["high"]), str(row["low"])


def add_accessible_supplement(speaker_df: pd.DataFrame, fomc_df: pd.DataFrame, seed_tsv: Path) -> tuple[pd.DataFrame, list[str]]:
    if not seed_tsv.exists():
        return speaker_df, [f"seed TSV missing: {seed_tsv}"]
    seed = pd.read_csv(seed_tsv, sep="\t")
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for _, rec in seed.iterrows():
        url = str(rec["url"]).strip()
        participant = str(rec["participant"]).strip()
        try:
            art = fetch_article(url)
            if not art.date_ymd:
                raise ValueError("could not infer date")
            ref, decision, high, low = nearest_fomc_label(art.date_ymd, fomc_df)
            doc = f"{art.title}. {art.text}".strip() if art.title else art.text
            rows.append(
                {
                    "fomc-ref-date": ref,
                    "date": art.date_ymd,
                    "decision": decision,
                    "high": high,
                    "low": low,
                    "domain": art.domain,
                    "participant": participant,
                    "document": doc,
                    "word_count": int(len(doc.split())),
                }
            )
        except Exception as e:
            failures.append(f"{url} :: {e}")
    if not rows:
        return speaker_df, failures
    supplement = pd.DataFrame(rows, columns=SPEAKER_COLUMNS)
    supplement = supplement.drop_duplicates(subset=["date", "domain", "participant", "document"], keep="first")
    merged = pd.concat([speaker_df, supplement], axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "domain", "participant", "document"], keep="first")
    merged = merged.sort_values(["date", "domain", "participant"], kind="mergesort").reset_index(drop=True)
    return merged[SPEAKER_COLUMNS], failures


def _print_distribution(title: str, s: pd.Series) -> None:
    print(f"\n{title}")
    vc = s.astype(str).value_counts(dropna=False)
    total = int(vc.sum())
    for k, v in vc.items():
        pct = 100.0 * float(v) / total if total else 0.0
        print(f"  {k!r}: {int(v)} ({pct:.1f}%)")


def _fomc_quality_masks(fomc_df: pd.DataFrame) -> dict[str, pd.Series]:
    doc_s = fomc_df["document"].astype(str)
    type_s = fomc_df["type"].astype(str) if "type" in fomc_df.columns else pd.Series(["statement"] * len(fomc_df), index=fomc_df.index)
    statement_mask = type_s.eq("statement")
    narrative_fallback = doc_s.str.contains(
        r"^FOMC (?:statement|placeholder) for \d{8}:",
        case=False,
        regex=True,
    )
    fred_labels = doc_s.str.contains(r"labels from FRED", case=False, regex=True)
    forward_filled = doc_s.str.contains(r"forward-filled", case=False, regex=True)
    parse_error = doc_s.str.contains(r"text unavailable or unparseable|parse error|HTML unavailable after URL", case=False, regex=True)
    true_stub = statement_mask & narrative_fallback
    return {
        "statement_mask": statement_mask,
        "narrative_fallback": narrative_fallback,
        "fred_labels": fred_labels,
        "forward_filled": forward_filled,
        "parse_error": parse_error,
        "true_stub": true_stub,
    }


def print_summary(fomc_df: pd.DataFrame, speaker_df: pd.DataFrame, failures: list[str], *, bis_added: int) -> None:
    print(f"\nWrote {FOMC_PICKLE} ({len(fomc_df)} rows)")
    print(f"Wrote {SPEAKER_PICKLE} ({len(speaker_df)} rows)")
    print(f"Wrote {ORIGINAL_FOMC_PATH} and {ORIGINAL_SPEAKER_PATH}")
    print(f"Backup of previous canonical pair: {BACKUP_FOMC} / {BACKUP_SPEAKER}")

    masks = _fomc_quality_masks(fomc_df)
    print("\n=== Dataset Completeness (FOMC) ===")
    print(f"  Rows with substantial text (word_count >= 80): {int((fomc_df['word_count'] >= 80).sum())} / {len(fomc_df)}")
    print(f"  True placeholder/fallback statement rows: {int(masks['true_stub'].sum())}")
    print(f"  Statement rows labeled from FRED fallback: {int((masks['statement_mask'] & masks['fred_labels']).sum())}")
    print(f"  Offline forward-filled statement rows: {int((masks['statement_mask'] & masks['forward_filled']).sum())}")
    print(f"  Type counts: {fomc_df['type'].astype(str).value_counts().to_dict()}")

    print("\n=== Dataset Completeness (Speaker) ===")
    print(f"  Total rows: {len(speaker_df)}")
    print(f"  Domain counts (top 12): {speaker_df['domain'].astype(str).value_counts().head(12).to_dict()}")
    if bis_added:
        print(f"  BIS gap-filler rows added: {bis_added}")
    if failures:
        print(f"  Accessible supplement fetch failures: {len(failures)}")

    post2020_fomc = fomc_df[fomc_df["date"].astype(str) >= "20200101"]
    post2020_speaker = speaker_df[speaker_df["date"].astype(str) >= "20200101"]
    print("\n=== Post-2020 Audit ===")
    print(f"  FOMC type counts: {post2020_fomc['type'].astype(str).value_counts().to_dict()}")
    print(f"  Speaker domain counts (top 12): {post2020_speaker['domain'].astype(str).value_counts().head(12).to_dict()}")

    print("\n=== ML Label Distribution (decision) ===")
    _print_distribution("fomc_doc.pkl — decision", fomc_df["decision"])
    _print_distribution("speaker_doc.pkl — decision", speaker_df["decision"])
    _print_distribution(
        "combined (FOMC + speaker) — decision",
        pd.concat([fomc_df["decision"], speaker_df["decision"]], ignore_index=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh all dataset files from cache/network and public legacy sources.")
    parser.add_argument("--offline", action="store_true", help="No HTTP: rely on local cache and already-downloaded originals.")
    parser.add_argument("--min-year", type=int, default=2015)
    parser.add_argument("--max-year", type=int, default=2026)
    parser.add_argument("--delay", type=float, default=0.35, help="Seconds between Fed HTTP GETs when fetching.")
    parser.add_argument("--no-fred", action="store_true", help="Do not download FRED series.")
    parser.add_argument("--refresh-public", action="store_true", help="Redownload public original FedNLP pickles.")
    parser.add_argument("--skip-bis-supplement", action="store_true", help="Do not add BIS Fed speech gap-fillers.")
    parser.add_argument("--skip-accessible-supplement", action="store_true", help="Do not add curated CNBC/Fox supplement.")
    args = parser.parse_args()

    backup_current_pair()

    min_ymd = f"{args.min_year:04d}0101"
    max_ymd = f"{args.max_year:04d}1231"
    cache_dir = DATA_DIR / ".cache" / "fed_html"
    if not cache_dir.is_dir():
        raise SystemExit(f"Missing cache directory: {cache_dir}")

    calendar_path = cache_dir / "_monetarypolicy_fomccalendars.htm.html"
    json_meta = cache_dir / "_json_ne-speeches.json.html"

    fred_bounds = None
    if not args.offline and not args.no_fred:
        try:
            fred_bounds = load_target_range_frame()
            print(f"Loaded FRED bounds: {len(fred_bounds)} rows (DFEDTARL / DFEDTARU).")
        except Exception as e:
            print(f"WARN: FRED load failed ({e}); continuing without FRED fallback.", file=sys.stderr)

    session: FedHttpSession | None = None
    if not args.offline:
        session = FedHttpSession(cache_dir=cache_dir, delay_s=args.delay)

    extra_cal_html: str | None = None
    if session is not None:
        try:
            extra_cal_html = session.get_text(FOMC_CALENDAR_URL)
            print("Merged live FOMC calendar HTML into meeting-date set.")
        except Exception as e:
            print(f"WARN: could not fetch FOMC calendar ({e}); using file cache + static lists only.", file=sys.stderr)

    print("Building current Fed-side FOMC table…")
    current_fomc = build_fomc_dataframe(
        cache_dir=cache_dir,
        calendar_path=calendar_path if calendar_path.is_file() else None,
        min_ymd=min_ymd,
        max_ymd=max_ymd,
        session=session,
        fred_bounds=fred_bounds,
        offline=args.offline,
        extra_calendar_html=extra_cal_html,
    )
    current_fomc = _coerce_fomc(current_fomc)

    print("Building current Fed-side speaker table…")
    speaker_raw = build_speaker_dataframe(
        cache_dir=cache_dir,
        json_cache_path=json_meta,
        min_ymd=min_ymd,
        max_ymd=max_ymd,
        session=session,
        offline=args.offline,
    )
    print(f"  Fed speeches indexed: {len(speaker_raw)}")
    label_fomc = current_fomc[current_fomc["type"].astype(str) == "statement"].copy()
    current_speaker = merge_speakers_nearest_fomc(speaker_raw, label_fomc)
    current_speaker = _coerce_speaker(current_speaker)

    if args.offline and (not ORIGINAL_FOMC_PATH.exists() or not ORIGINAL_SPEAKER_PATH.exists()):
        raise SystemExit("Offline mode needs existing data/fomc_doc_original.pkl and data/speaker_doc_original.pkl")

    download_if_needed(PUBLIC_FOMC_URL, ORIGINAL_FOMC_PATH, refresh=args.refresh_public and not args.offline)
    download_if_needed(PUBLIC_SPEAKER_URL, ORIGINAL_SPEAKER_PATH, refresh=args.refresh_public and not args.offline)

    public_fomc = pd.read_pickle(ORIGINAL_FOMC_PATH)
    public_speaker = pd.read_pickle(ORIGINAL_SPEAKER_PATH)
    public_fomc = _coerce_fomc(public_fomc)
    public_speaker = _coerce_speaker(public_speaker)

    fomc = merge_public_fomc(current_fomc, public_fomc)
    speaker = merge_public_speaker(current_speaker, public_speaker)
    bis_added = 0
    if not args.skip_bis_supplement and not args.offline:
        bis_rows = add_bis_gap_fillers(
            speaker,
            min_year=args.min_year,
            max_year=args.max_year,
            cache_dir=DATA_DIR / ".cache" / "bis",
        )
        if not bis_rows.empty:
            for idx in bis_rows.index:
                ref, decision, high, low = nearest_fomc_label(str(bis_rows.at[idx, "date"]), label_fomc)
                bis_rows.at[idx, "fomc-ref-date"] = ref
                bis_rows.at[idx, "decision"] = decision
                bis_rows.at[idx, "high"] = high
                bis_rows.at[idx, "low"] = low
            bis_added = len(bis_rows)
            speaker = pd.concat([speaker, bis_rows], axis=0, ignore_index=True)
            speaker = speaker.drop_duplicates(subset=["date", "domain", "participant", "document"], keep="first")
            speaker = speaker.sort_values(["date", "domain", "participant"], kind="mergesort").reset_index(drop=True)
    failures: list[str] = []
    if not args.skip_accessible_supplement and not args.offline:
        speaker, failures = add_accessible_supplement(speaker, label_fomc, SEED_TSV)

    validate_frames(fomc.drop(columns=["type"], errors="ignore"), speaker)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fomc.to_pickle(FOMC_PICKLE)
    speaker.to_pickle(SPEAKER_PICKLE)

    print("\nVerifying loaders…")
    load_fomc(str(FOMC_PICKLE))
    load_speaker(str(SPEAKER_PICKLE))
    load_fomc(str(ORIGINAL_FOMC_PATH))
    load_speaker(str(ORIGINAL_SPEAKER_PATH))
    print("OK: canonical and original pairs passed schema checks.")

    print_summary(fomc, speaker, failures, bis_added=bis_added)
    if failures:
        print("\nSkipped accessible supplement URLs:")
        for msg in failures:
            print(f"  - {msg}")
    print("\nNext month, rerun exactly:")
    print("  python scripts/rebuild_database.py")


if __name__ == "__main__":
    main()
