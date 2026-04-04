#!/usr/bin/env python3
"""
Build speaker_doc_fed_extended.pkl: Fed-system rows from the original speaker pickle
plus new speeches from federalreserve.gov (JSON index + HTML body).

- Copies speaker_doc.pkl -> speaker_doc_backup.pkl once if missing.
- Keeps date and fomc-ref-date as YYYYMMDD strings (never datetime) after concat.
- Joins FOMC labels via next meeting date >= speech date; policy from legacy
  pickle rows or parsed official press releases (regex + BeautifulSoup).

Run from repo root with venv activated:

  python scripts/extend_speaker_federalreserve.py

Options: --min-date, --max-date, --delay, --limit (debug), --no-backup
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ingest import SPEAKER_COLUMNS  # noqa: E402
from scraping.fed_official import (  # noqa: E402
    BASE,
    FedHttpSession,
    extract_speech_body,
    fetch_meeting_policy,
    fetch_speeches_metadata,
    scrape_minutes_meeting_dates,
    speech_date_to_ymd,
)

DATA_DIR = ROOT / "data"
SPEAKER_SRC = DATA_DIR / "speaker_doc.pkl"
SPEAKER_BACKUP = DATA_DIR / "speaker_doc_backup.pkl"
SPEAKER_OUT = DATA_DIR / "speaker_doc_fed_extended.pkl"
CACHE_DIR = DATA_DIR / ".cache" / "fed_html"

EXTRA_2020_MEETINGS = frozenset({"20200916", "20201105", "20201216"})


def is_fed_system_domain(domain: str) -> bool:
    d = str(domain).lower().strip()
    if "federalreserve.gov" in d:
        return True
    if d.endswith("fed.org") or ".fed.org" in d:
        return True
    if "frb.org" in d:
        return True
    if "frbatlanta.org" in d or "frbsf.org" in d:
        return True
    return False


def legacy_meeting_policy(speaker_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    u = speaker_df.drop_duplicates(subset=["fomc-ref-date"], keep="first")
    out: dict[str, dict[str, str]] = {}
    for _, row in u.iterrows():
        k = str(row["fomc-ref-date"]).strip()
        out[k] = {
            "decision": str(row["decision"]),
            "high": str(row["high"]),
            "low": str(row["low"]),
        }
    return out


def all_meeting_dates(legacy_keys: set[str], session: FedHttpSession) -> list[str]:
    scraped = scrape_minutes_meeting_dates(session)
    merged = set(legacy_keys) | set(scraped) | set(EXTRA_2020_MEETINGS)
    return sorted(merged)


def next_meeting_on_or_after(
    speech_ymd: str,
    meetings_sorted: list[str],
    *,
    carry_forward_last: bool = True,
) -> str | None:
    """Map speech date to the next scheduled FOMC meeting date in the list (YYYYMMDD strings).

    If the speech is after the last listed meeting and carry_forward_last is True, use that
    last meeting so labels stay within the chosen horizon (e.g. through end-2025).
    """
    if not meetings_sorted:
        return None
    last_m = meetings_sorted[-1]
    for m in meetings_sorted:
        if m >= speech_ymd:
            return m
    if carry_forward_last and speech_ymd > last_m:
        return last_m
    return None


def ensure_yyyymmdd_str(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if hasattr(val, "strftime"):
        return val.strftime("%Y%m%d")  # type: ignore[union-attr]
    s = str(val).strip()
    if re.fullmatch(r"\d{8}", s):
        return s
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return s
    return pd.Timestamp(ts).strftime("%Y%m%d")


def coerce_speaker_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["date"].map(ensure_yyyymmdd_str)
    out["fomc-ref-date"] = out["fomc-ref-date"].map(ensure_yyyymmdd_str)
    for c in ("decision", "high", "low", "domain", "participant", "document"):
        out[c] = out[c].astype(str)
    out["word_count"] = out["word_count"].astype("int64")
    return out


def validate_speaker(df: pd.DataFrame) -> None:
    missing = [c for c in SPEAKER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    bad_dates = ~df["date"].astype(str).str.match(r"^\d{8}$", na=False)
    if bad_dates.any():
        raise ValueError(f"Non-YYYYMMDD date values: {df.loc[bad_dates, 'date'].head()}")
    bad_ref = ~df["fomc-ref-date"].astype(str).str.match(r"^\d{8}$", na=False)
    if bad_ref.any():
        raise ValueError(f"Non-YYYYMMDD fomc-ref-date: {df.loc[bad_ref, 'fomc-ref-date'].head()}")


def build_policy_map(
    meetings_sorted: list[str],
    legacy: dict[str, dict[str, str]],
    session: FedHttpSession,
) -> dict[str, dict[str, str]]:
    policy: dict[str, dict[str, str]] = {}
    for m in meetings_sorted:
        if m in legacy:
            policy[m] = legacy[m]
            continue
        info = fetch_meeting_policy(session, m)
        if "error" in info:
            print(f"WARN: FOMC {m} — {info.get('error')}: {info.get('snippet', '')[:120]}", file=sys.stderr)
            continue
        policy[m] = {
            "decision": str(info["decision"]),
            "high": str(info["high"]),
            "low": str(info["low"]),
        }
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend Fed-system speaker pickle from federalreserve.gov")
    parser.add_argument(
        "--min-date",
        default="20200717",
        help="Minimum speech date YYYYMMDD (inclusive). Default skips corpus overlap through 20200716.",
    )
    parser.add_argument("--max-date", default="20251231", help="Last speech date YYYYMMDD (inclusive)")
    parser.add_argument(
        "--max-fomc-ref",
        default="20251231",
        help="Drop FOMC meetings after this YYYYMMDD when mapping labels (default keeps corpus through end of 2025).",
    )
    parser.add_argument("--delay", type=float, default=0.35, help="Seconds between HTTP GETs")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only fetch this many new speeches (debug)")
    parser.add_argument("--no-backup", action="store_true", help="Do not create speaker_doc_backup.pkl")
    parser.add_argument("--no-cache", action="store_true", help="Do not write HTML cache under data/.cache/")
    args = parser.parse_args()

    if not SPEAKER_SRC.exists():
        raise SystemExit(f"Missing {SPEAKER_SRC}")

    if not args.no_backup and not SPEAKER_BACKUP.exists():
        shutil.copy2(SPEAKER_SRC, SPEAKER_BACKUP)
        print(f"Wrote backup: {SPEAKER_BACKUP}")

    speaker_full = pd.read_pickle(SPEAKER_SRC)
    base_fed = speaker_full[speaker_full["domain"].map(is_fed_system_domain)].copy()

    legacy = legacy_meeting_policy(speaker_full)
    cache = None if args.no_cache else CACHE_DIR
    session = FedHttpSession(cache_dir=cache, delay_s=args.delay)

    max_ref = args.max_fomc_ref.strip()
    meetings = [m for m in all_meeting_dates(set(legacy.keys()), session) if m <= max_ref]
    print(f"Meeting dates tracked (<= {max_ref}): {len(meetings)} (legacy + scraped + 2020 gap fill)")
    policy_map = build_policy_map(meetings, legacy, session)
    print(f"Policy rows resolved: {len(policy_map)}")

    meta = fetch_speeches_metadata(session)
    min_d = args.min_date.strip()
    max_d = args.max_date.strip()

    new_rows: list[dict] = []
    seen_slug: set[str] = set()

    for row in meta:
        if args.limit and len(new_rows) >= args.limit:
            break
        link = str(row.get("l") or "")
        if not link.startswith("/newsevents/speech/"):
            continue
        slug = Path(link).name
        if slug in seen_slug:
            continue
        ymd = speech_date_to_ymd(str(row.get("d") or ""))
        if ymd <= "20200716" or ymd > max_d:
            continue
        if ymd < min_d:
            continue

        url = BASE + link
        try:
            html = session.get_text(url)
        except Exception as e:
            print(f"WARN: speech fetch failed {url}: {e}", file=sys.stderr)
            continue
        body = extract_speech_body(html)
        if len(body) < 80:
            print(f"WARN: short body for {slug}", file=sys.stderr)
            continue
        ref = next_meeting_on_or_after(ymd, meetings)
        if not ref or ref not in policy_map:
            print(f"WARN: no FOMC policy for speech {ymd} -> {ref}", file=sys.stderr)
            continue
        pol = policy_map[ref]
        participant = str(row.get("s") or "").strip()
        title = str(row.get("t") or "").strip()
        doc = body if not title else f"{title}. {body}"
        new_rows.append(
            {
                "fomc-ref-date": ref,
                "date": ymd,
                "decision": pol["decision"],
                "high": pol["high"],
                "low": pol["low"],
                "domain": "www.federalreserve.gov",
                "participant": participant,
                "document": doc,
                "word_count": int(len(doc.split())),
            }
        )
        seen_slug.add(slug)

    new_df = pd.DataFrame(new_rows, columns=SPEAKER_COLUMNS)
    combined = pd.concat([base_fed, new_df], axis=0, ignore_index=True)
    combined = coerce_speaker_frame(combined)
    validate_speaker(combined)

    SPEAKER_OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(SPEAKER_OUT)
    print(
        f"Wrote {SPEAKER_OUT} rows={len(combined)} "
        f"(fed_base={len(base_fed)} + new_fed_speeches={len(new_df)})"
    )


if __name__ == "__main__":
    main()
