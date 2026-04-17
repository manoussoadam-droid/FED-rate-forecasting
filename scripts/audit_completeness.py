#!/usr/bin/env python3
"""CLI: verify the Parquet store contains the expected FOMC documents.

For each scheduled FOMC meeting in the requested year range, checks that a
``statement`` + ``minutes`` (+ ``press-conference`` since 2019) row is
present with non-empty text and no disqualifying ``quality_flags``.

Exit codes:
  * 0 — all expected documents present
  * 2 — at least one expected document is missing or flagged
  * 1 — internal error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import DATA_DIR  # noqa: E402
from rebuild.audit import format_markdown, report_completeness  # noqa: E402


def _calendar_path() -> Path | None:
    cal = DATA_DIR / ".cache" / "fed_html" / "_monetarypolicy_fomccalendars.htm.html"
    return cal if cal.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-year", type=int, default=2020)
    ap.add_argument("--max-year", type=int, default=None)
    ap.add_argument("--format", choices=("md", "csv", "table"), default="table")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to write the report")
    args = ap.parse_args()

    max_year = args.max_year or pd.Timestamp.utcnow().year
    cal_path = _calendar_path()
    frames: list[pd.DataFrame] = []
    for y in range(args.min_year, max_year + 1):
        df = report_completeness(y, calendar_html_path=cal_path)
        if not df.empty:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if combined.empty:
        print("No scheduled meetings found in range.", file=sys.stderr)
        return 2

    title = f"FOMC coverage {args.min_year}-{max_year}"
    if args.format == "md":
        rendered = format_markdown(combined, title=title)
    elif args.format == "csv":
        rendered = combined.to_csv(index=False)
    else:
        cols = [
            "date",
            "statement",
            "minutes",
            "press_conference",
            "press_conference_expected",
            "complete",
        ]
        rendered = combined[cols].to_string(index=False)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)

    incomplete = combined[~combined["complete"]]
    if not incomplete.empty:
        missing = len(incomplete)
        total = len(combined)
        print(
            f"\n{missing}/{total} meetings incomplete (exit 2).",
            file=sys.stderr,
        )
        return 2

    print(f"\nAll {len(combined)} scheduled meetings are complete.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
