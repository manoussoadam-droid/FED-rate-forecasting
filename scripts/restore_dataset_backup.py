#!/usr/bin/env python3
"""Restore the canonical dataset pair from the latest automatic backup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import FOMC_PICKLE, SPEAKER_PICKLE  # noqa: E402

BACKUP_DIR = ROOT / "artifacts" / "data_backups"
BACKUP_FOMC = BACKUP_DIR / "fomc_doc.last_backup.pkl"
BACKUP_SPEAKER = BACKUP_DIR / "speaker_doc.last_backup.pkl"


def main() -> None:
    if not BACKUP_FOMC.exists() or not BACKUP_SPEAKER.exists():
        raise SystemExit("Missing latest dataset backup.")
    FOMC_PICKLE.write_bytes(BACKUP_FOMC.read_bytes())
    SPEAKER_PICKLE.write_bytes(BACKUP_SPEAKER.read_bytes())
    print(f"Restored: {FOMC_PICKLE}")
    print(f"Restored: {SPEAKER_PICKLE}")


if __name__ == "__main__":
    main()
