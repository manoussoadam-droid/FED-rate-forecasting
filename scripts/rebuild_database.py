#!/usr/bin/env python3
"""Refresh all dataset files used by the project.

This is the one script to rerun when new data is available.

Outputs in `data/`:
- `data/parquet/fomc/`    — canonical FOMC corpus (Parquet, partitioned by year)
- `data/parquet/speaker/` — canonical speaker corpus (Parquet, partitioned by year)
- `data/audit.sqlite`     — ingestion log + document audit + fetch errors
- `data/fomc_doc_original.parquet`    — cached FedNLP seed (FOMC)
- `data/speaker_doc_original.parquet` — cached FedNLP seed (speaker)
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import DATA_DIR, ERRORS_CSV  # noqa: E402
from core.ingest import FOMC_COLUMNS_LEGACY, SPEAKER_COLUMNS, load_fomc, load_speaker  # noqa: E402
from core.repository import DocumentRepository  # noqa: E402
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

# Public FedNLP seed data (pickle format from external source — read in-memory,
# never saved to disk as pickle; cached locally as Parquet after first download).
PUBLIC_FOMC_URL = "https://raw.githubusercontent.com/usydnlp/FedNLP/main/resources/fomc_doc.pkl"
PUBLIC_SPEAKER_URL = "https://raw.githubusercontent.com/usydnlp/FedNLP/main/resources/speaker_doc.pkl"
ORIGINAL_FOMC_PARQUET = DATA_DIR / "fomc_doc_original.parquet"
ORIGINAL_SPEAKER_PARQUET = DATA_DIR / "speaker_doc_original.parquet"
SEED_TSV = ROOT / "scripts" / "nonfed_accessible_seed_urls.tsv"
BACKUP_DIR = ROOT / "artifacts" / "data_backups"


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


def backup_parquet_snapshot() -> None:
    """Copy the current Parquet store into artifacts/data_backups/ for rollback."""
    from core.config import FOMC_PARQUET_DIR, SPEAKER_PARQUET_DIR

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src, name in [
        (FOMC_PARQUET_DIR, "fomc_parquet_backup"),
        (SPEAKER_PARQUET_DIR, "speaker_parquet_backup"),
    ]:
        if src.exists() and any(src.rglob("*.parquet")):
            dest = BACKUP_DIR / name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)


def _write_run_errors_csv(run_id: str, failures: list[str]) -> None:
    """Append supplemental fetch failures for the current run to ``data/errors.csv``."""
    if not failures:
        return
    ERRORS_CSV.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not ERRORS_CSV.exists()
    with ERRORS_CSV.open("a", encoding="utf-8") as fh:
        if header_needed:
            fh.write("run_id,kind,message\n")
        for msg in failures:
            safe = str(msg).replace('"', "'").replace("\n", " ").replace("\r", " ")
            fh.write(f'{run_id},supplement,"{safe}"\n')


def _fetch_fednlp_parquet(url: str, parquet_path: Path, *, refresh: bool) -> pd.DataFrame:
    """Download a FedNLP pickle URL, read it in-memory, cache as Parquet, return DataFrame.

    The external FedNLP repo only publishes pickle files.  We read the bytes
    directly into ``io.BytesIO`` so no ``.pkl`` file is ever written to disk;
    the result is persisted locally as Parquet for subsequent runs.
    """
    if parquet_path.exists() and not refresh:
        return pd.read_parquet(parquet_path)
    print(f"  Downloading {url} …")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_pickle(io.BytesIO(r.content))  # external-source pickle, read in-memory
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return df


_FOMC_EXTRAS = ("source_url", "quality_flags", "labels_from_fred", "parser_version")
_SPEAKER_EXTRAS = ("source_url", "quality_flags", "parser_version", "alignment_rule")


def _coerce_fomc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "type" not in out.columns:
        out["type"] = "statement"
    out["date"] = out["date"].map(ensure_yyyymmdd)
    for c in ("type", "decision", "high", "low", "document"):
        out[c] = out[c].astype(str)
    out["word_count"] = out["word_count"].astype("int64")
    base = list(FOMC_COLUMNS_LEGACY)
    extras = [c for c in _FOMC_EXTRAS if c in out.columns]
    return out[base + extras]


def _coerce_speaker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["date"].map(ensure_yyyymmdd)
    out["fomc-ref-date"] = out["fomc-ref-date"].map(ensure_yyyymmdd)
    for c in ("decision", "high", "low", "domain", "participant", "document"):
        out[c] = out[c].astype(str)
    out["word_count"] = out["word_count"].astype("int64")
    base = list(SPEAKER_COLUMNS)
    extras = [c for c in _SPEAKER_EXTRAS if c in out.columns]
    return out[base + extras]


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
    print(f"\nFOMC corpus: {len(fomc_df)} rows")
    print(f"Speaker corpus: {len(speaker_df)} rows")
    print(f"FedNLP seed cache: {ORIGINAL_FOMC_PARQUET} / {ORIGINAL_SPEAKER_PARQUET}")
    print(f"Parquet backup: {BACKUP_DIR / 'fomc_parquet_backup'} / {BACKUP_DIR / 'speaker_parquet_backup'}")

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
    _print_distribution("fomc corpus — decision", fomc_df["decision"])
    _print_distribution("speaker corpus — decision", speaker_df["decision"])
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
    parser.add_argument("--refresh-public", action="store_true", help="Redownload public original FedNLP seed data (ignores local Parquet cache).")
    parser.add_argument("--skip-bis-supplement", action="store_true", help="Do not add BIS Fed speech gap-fillers.")
    parser.add_argument("--skip-accessible-supplement", action="store_true", help="Do not add curated CNBC/Fox supplement.")
    parser.add_argument(
        "--force-refresh-stubs",
        action="store_true",
        help=(
            "Re-fetch URLs whose audited quality_flags include no_text / "
            "pdf_unreadable / html_structure_unknown, bypassing the on-disk "
            "cache so the upgraded parser can replace legacy stubs."
        ),
    )
    parser.add_argument(
        "--force-refresh-all",
        action="store_true",
        help="Bypass the on-disk cache for every request this run.",
    )
    args = parser.parse_args()

    backup_parquet_snapshot()

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

    refresh_urls: set[str] = set()
    if args.force_refresh_stubs and not args.offline:
        try:
            pre_repo = DocumentRepository()
            refresh_urls = {u for u, _ in pre_repo.stub_urls()}
            print(f"--force-refresh-stubs: {len(refresh_urls)} URLs will bypass cache")
        except Exception as e:
            print(f"WARN: could not load stub URLs from audit DB ({e})", file=sys.stderr)

    session: FedHttpSession | None = None
    if not args.offline:
        session = FedHttpSession(
            cache_dir=cache_dir,
            delay_s=args.delay,
            force_refresh=args.force_refresh_all,
            refresh_urls=refresh_urls,
        )

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

    if args.offline and (not ORIGINAL_FOMC_PARQUET.exists() or not ORIGINAL_SPEAKER_PARQUET.exists()):
        raise SystemExit(
            "Offline mode requires a previously cached FedNLP seed.\n"
            f"  Expected: {ORIGINAL_FOMC_PARQUET}\n"
            f"  Expected: {ORIGINAL_SPEAKER_PARQUET}\n"
            "Run once without --offline to download and cache the FedNLP seed data."
        )

    refresh_public = args.refresh_public and not args.offline
    public_fomc = _fetch_fednlp_parquet(PUBLIC_FOMC_URL, ORIGINAL_FOMC_PARQUET, refresh=refresh_public)
    public_speaker = _fetch_fednlp_parquet(PUBLIC_SPEAKER_URL, ORIGINAL_SPEAKER_PARQUET, refresh=refresh_public)
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

    # Canonical write: Parquet (partitioned by year) + SQLite audit trail.
    repo = DocumentRepository()
    run_id = repo.start_run()
    rows_fomc = 0
    rows_speaker = 0
    try:
        rows_fomc = repo.write_fomc(fomc)
        rows_speaker = repo.write_speaker(speaker)
        print(f"\nWrote Parquet: fomc={rows_fomc}, speaker={rows_speaker}")
        print(f"Audit DB: {repo.audit_db}")
        _write_run_errors_csv(run_id, failures)
    finally:
        repo.finish_run(
            run_id,
            rows_fomc=rows_fomc,
            rows_speaker=rows_speaker,
            errors_csv_path=str(ERRORS_CSV) if failures else "",
        )

    print("\nVerifying loaders…")
    fomc_check = load_fomc()
    speaker_check = load_speaker()
    print(f"OK: loaders verified — fomc={len(fomc_check)} rows, speaker={len(speaker_check)} rows")

    print_summary(fomc, speaker, failures, bis_added=bis_added)
    if failures:
        print("\nSkipped accessible supplement URLs:")
        for msg in failures:
            print(f"  - {msg}")
    print("\nTo refresh next month, rerun:")
    print("  python scripts/rebuild_database.py")


if __name__ == "__main__":
    main()
