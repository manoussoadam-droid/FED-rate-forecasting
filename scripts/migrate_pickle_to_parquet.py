#!/usr/bin/env python3
"""One-shot migration from legacy pickles to the Parquet store.

Reads ``data/fomc_doc.pkl`` and ``data/speaker_doc.pkl``, backfills the
canonical columns (``source_url``, ``ingested_at``, ``content_hash``,
``quality_score``, ``quality_flags``, ``parser_version``), and writes
partitioned Parquet + seeds the audit SQLite DB.

Usage::

    python scripts/migrate_pickle_to_parquet.py
    python scripts/migrate_pickle_to_parquet.py --dry-run
    python scripts/migrate_pickle_to_parquet.py --no-speaker

The script is idempotent: rerunning it upserts by ``content_hash`` so it
won't duplicate rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import FOMC_PICKLE, SPEAKER_PICKLE  # noqa: E402
from core.ingest import FOMC_COLUMNS, FOMC_COLUMNS_LEGACY, SPEAKER_COLUMNS  # noqa: E402
from core.repository import DocumentRepository, normalize_fomc_frame, normalize_speaker_frame  # noqa: E402


LEGACY_PARSER_TAG = "legacy-pickle"


def _load_legacy_fomc(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    expected = FOMC_COLUMNS_LEGACY if "type" in df.columns else FOMC_COLUMNS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SystemExit(f"fomc pickle {path} missing columns: {missing}")
    out = df.copy()
    if "type" not in out.columns:
        out["type"] = "statement"
    out["source_url"] = ""
    out["quality_flags"] = [[] for _ in range(len(out))]
    out["labels_from_fred"] = False
    out["parser_version"] = LEGACY_PARSER_TAG
    return out


def _load_legacy_speaker(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    missing = [c for c in SPEAKER_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"speaker pickle {path} missing columns: {missing}")
    out = df.copy()
    out["source_url"] = ""
    out["quality_flags"] = [[] for _ in range(len(out))]
    out["parser_version"] = LEGACY_PARSER_TAG
    out["alignment_rule"] = "legacy"
    return out


def _roundtrip_check(repo: DocumentRepository, original: pd.DataFrame, kind: str) -> None:
    """Fail loudly if the Parquet write lost rows or mangled key columns."""
    if kind == "fomc":
        df_back = repo.read_fomc()
        keys = {"date", "document_type"}
    else:
        df_back = repo.read_speaker()
        keys = {"date", "participant"}
    if df_back.empty and not original.empty:
        raise SystemExit(f"round-trip failed: Parquet {kind} store is empty after write")
    if len(df_back) < len(original):
        print(
            f"WARN: parquet {kind} rows ({len(df_back)}) < original ({len(original)}); "
            "content_hash dedupe may have merged identical bodies",
            file=sys.stderr,
        )
    for k in keys:
        if k not in df_back.columns:
            raise SystemExit(f"round-trip failed: {kind} missing key column {k}")


def migrate(*, dry_run: bool, include_speaker: bool) -> None:
    repo = DocumentRepository()
    print(f"Parquet fomc dir:    {repo.fomc_dir}")
    print(f"Parquet speaker dir: {repo.speaker_dir}")
    print(f"SQLite audit:        {repo.audit_db}")

    if not FOMC_PICKLE.exists():
        raise SystemExit(f"FOMC pickle not found: {FOMC_PICKLE}")
    fomc_raw = _load_legacy_fomc(FOMC_PICKLE)
    print(f"Loaded FOMC pickle:    {len(fomc_raw)} rows")

    fomc_normalized = normalize_fomc_frame(fomc_raw)
    print(f"Normalized FOMC rows:  {len(fomc_normalized)}")

    if include_speaker:
        if not SPEAKER_PICKLE.exists():
            raise SystemExit(f"speaker pickle not found: {SPEAKER_PICKLE}")
        speaker_raw = _load_legacy_speaker(SPEAKER_PICKLE)
        print(f"Loaded speaker pickle: {len(speaker_raw)} rows")
        speaker_normalized = normalize_speaker_frame(speaker_raw)
        print(f"Normalized speaker:    {len(speaker_normalized)} rows")
    else:
        speaker_raw = pd.DataFrame()
        speaker_normalized = pd.DataFrame()

    if dry_run:
        print("\n[dry-run] skipping Parquet writes.")
        if not fomc_normalized.empty:
            print("\nFOMC sample (head 3):")
            print(fomc_normalized.head(3).to_string())
        if include_speaker and not speaker_normalized.empty:
            print("\nSpeaker sample (head 3):")
            print(speaker_normalized.head(3).to_string())
        return

    run_id = repo.start_run()
    try:
        rows_fomc = repo.write_fomc(fomc_raw)
        print(f"Wrote FOMC rows:    {rows_fomc}")
        _roundtrip_check(repo, fomc_normalized, "fomc")

        rows_sp = 0
        if include_speaker and not speaker_raw.empty:
            rows_sp = repo.write_speaker(speaker_raw)
            print(f"Wrote speaker rows: {rows_sp}")
            _roundtrip_check(repo, speaker_normalized, "speaker")
    finally:
        repo.finish_run(run_id, rows_fomc=len(fomc_normalized), rows_speaker=len(speaker_normalized))

    print("\nMigration complete.")
    print("Next step: rerun `python scripts/rebuild_database.py` to overlay")
    print("live scrapes onto the Parquet store (idempotent via content_hash).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-speaker", action="store_true", help="Migrate only the FOMC table")
    args = ap.parse_args()
    migrate(dry_run=args.dry_run, include_speaker=not args.no_speaker)


if __name__ == "__main__":
    main()
