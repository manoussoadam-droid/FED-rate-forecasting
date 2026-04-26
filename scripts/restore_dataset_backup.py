#!/usr/bin/env python3
"""Restore the canonical Parquet dataset from the latest automatic backup."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import FOMC_PARQUET_DIR, SPEAKER_PARQUET_DIR  # noqa: E402

BACKUP_DIR = ROOT / "artifacts" / "data_backups"
BACKUP_FOMC = BACKUP_DIR / "fomc_parquet_backup"
BACKUP_SPEAKER = BACKUP_DIR / "speaker_parquet_backup"


def main() -> None:
    missing = [p for p in (BACKUP_FOMC, BACKUP_SPEAKER) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing Parquet backup(s): {missing}\n"
            "Run `python scripts/rebuild_database.py` at least once to create a backup."
        )

    for src, dest in [(BACKUP_FOMC, FOMC_PARQUET_DIR), (BACKUP_SPEAKER, SPEAKER_PARQUET_DIR)]:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"Restored: {dest}")


if __name__ == "__main__":
    main()
