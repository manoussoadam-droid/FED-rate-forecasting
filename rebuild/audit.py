"""Completeness auditing for the FOMC corpus.

For each scheduled FOMC meeting in a given year, verify that the canonical
Parquet store contains a *statement*, *minutes*, and (post-2019) a
*press-conference* row with non-empty ``text_content`` and no disqualifying
``quality_flags``.

The output is a pandas DataFrame suitable for printing or writing to the
audit SQLite DB; :func:`report_is_complete` collapses it to a single
boolean for CI gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.repository import DocumentRepository
from rebuild.meetings import all_meeting_dates

# Press conferences occur at *every* FOMC meeting from January 2019 onward.
PRESS_CONF_START_YMD = "20190101"

# Quality flags that disqualify a row from "present" status.
_BLOCKING_FLAGS = frozenset({"no_text", "pdf_unreadable", "html_structure_unknown"})


@dataclass
class MeetingCoverage:
    date: str
    statement: bool
    minutes: bool
    press_conference: bool
    press_conference_expected: bool
    statement_flags: list[str]
    minutes_flags: list[str]
    press_conference_flags: list[str]

    @property
    def complete(self) -> bool:
        need_pc = self.press_conference if self.press_conference_expected else True
        return self.statement and self.minutes and need_pc


def _row_present(rows: pd.DataFrame) -> tuple[bool, list[str]]:
    if rows.empty:
        return False, []
    # Pick the longest text we have for this meeting/doc type.
    rows = rows.copy()
    rows["_wc"] = rows["text_content"].astype(str).map(lambda t: len(t.split()))
    rows = rows.sort_values("_wc", ascending=False)
    top = rows.iloc[0]
    flags = list(top.get("quality_flags") or [])
    text = str(top.get("text_content") or "")
    blocked = bool(_BLOCKING_FLAGS.intersection(flags))
    present = bool(text.strip()) and not blocked
    return present, flags


def report_completeness(
    year: int,
    *,
    calendar_html_path: Path | None = None,
    repo: DocumentRepository | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with one row per scheduled meeting in ``year``.

    Columns:
      - ``date``, ``year``
      - ``statement``, ``minutes``, ``press_conference`` booleans
      - ``press_conference_expected`` bool
      - ``statement_flags``, ``minutes_flags``, ``press_conference_flags`` (list[str])
      - ``complete`` bool (convenience)
    """

    repo = repo or DocumentRepository()
    min_ymd = f"{year:04d}0101"
    max_ymd = f"{year:04d}1231"
    meetings = all_meeting_dates(
        calendar_html_path,
        min_ymd=min_ymd,
        max_ymd=max_ymd,
    )
    fomc = repo.read_fomc(year_range=(year, year))
    records: list[MeetingCoverage] = []
    for ymd in meetings:
        sub = fomc[fomc["date"].astype(str) == ymd] if not fomc.empty else pd.DataFrame()
        stmt_rows = sub[sub["document_type"] == "statement"] if not sub.empty else pd.DataFrame()
        min_rows = sub[sub["document_type"] == "minutes"] if not sub.empty else pd.DataFrame()
        pc_rows = sub[sub["document_type"] == "press-conference"] if not sub.empty else pd.DataFrame()
        stmt_ok, stmt_flags = _row_present(stmt_rows)
        min_ok, min_flags = _row_present(min_rows)
        pc_ok, pc_flags = _row_present(pc_rows)
        expect_pc = ymd >= PRESS_CONF_START_YMD
        records.append(
            MeetingCoverage(
                date=ymd,
                statement=stmt_ok,
                minutes=min_ok,
                press_conference=pc_ok,
                press_conference_expected=expect_pc,
                statement_flags=stmt_flags,
                minutes_flags=min_flags,
                press_conference_flags=pc_flags,
            )
        )

    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "year": int(r.date[:4]),
                "statement": r.statement,
                "minutes": r.minutes,
                "press_conference": r.press_conference,
                "press_conference_expected": r.press_conference_expected,
                "statement_flags": r.statement_flags,
                "minutes_flags": r.minutes_flags,
                "press_conference_flags": r.press_conference_flags,
                "complete": r.complete,
            }
            for r in records
        ]
    )
    return df


def report_is_complete(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    return bool(df["complete"].all())


def format_markdown(df: pd.DataFrame, *, title: str = "Completeness report") -> str:
    """Render the DataFrame as a compact markdown table for CI logs."""
    if df.empty:
        return f"# {title}\n\n_No meetings in range._\n"
    rows = ["| date | statement | minutes | press-conf | expected | flags |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        flags: list[str] = []
        for key in ("statement_flags", "minutes_flags", "press_conference_flags"):
            val = r.get(key) or []
            if isinstance(val, list) and val:
                flags.extend(val)
        flag_cell = ",".join(sorted(set(flags))) or "-"
        rows.append(
            f"| {r['date']} "
            f"| {'yes' if r['statement'] else 'no'} "
            f"| {'yes' if r['minutes'] else 'no'} "
            f"| {'yes' if r['press_conference'] else 'no'} "
            f"| {'yes' if r['press_conference_expected'] else 'no'} "
            f"| {flag_cell} |"
        )
    return f"# {title}\n\n" + "\n".join(rows) + "\n"


__all__ = [
    "MeetingCoverage",
    "report_completeness",
    "report_is_complete",
    "format_markdown",
    "PRESS_CONF_START_YMD",
]
