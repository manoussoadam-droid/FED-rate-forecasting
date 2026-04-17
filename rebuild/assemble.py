"""Scan cache, build DataFrames, nearest FOMC meeting for speaker labels."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from core.ingest import FOMC_COLUMNS, FOMC_COLUMNS_LEGACY, SPEAKER_COLUMNS
from rebuild.extract import (
    extract_fomc_statement_body,
    extract_minutes_body,
    extract_pdf_text,
    parse_board_event_page,
)
from rebuild.fred_rates import infer_policy_from_fred
from rebuild.meetings import all_meeting_dates
from scraping.document_parser import (
    FLAG_HTTP_ERROR,
    FLAG_NO_TEXT,
    FLAG_PDF_UNREADABLE,
    FLAG_SHORT_BODY,
    PARSER_VERSION,
)
from scraping.fed_official import (
    BASE,
    FedHttpSession,
    SPEECHES_JSON,
    board_year_listing_url,
    meeting_minutes_urls,
    meeting_press_conference_urls,
    meeting_statement_urls,
    parse_monetary_statement,
    speech_date_to_ymd,
)


def _cache_key_from_url(url: str) -> str:
    return re.sub(r"[^\w.-]+", "_", urlparse(url).path)[:200]


_CACHE_SPEECH_RE = re.compile(
    r"_newsevents_speech_([a-z]+)(\d{8})([a-z])\.htm\.html$",
    re.I,
)
_BOARD_DOC_RE = re.compile(r"/newsevents/(speech|testimony)/([a-z0-9]+)\.htm$", re.I)
_YMD_IN_SLUG_RE = re.compile(r"(20\d{6})")


def _fetch_url_text(
    session: FedHttpSession | None,
    url: str,
    *,
    offline: bool,
    warn: bool = True,
) -> str | None:
    if session is None or offline:
        return None
    try:
        return session.get_text(url)
    except Exception as e:
        if warn:
            print(f"WARN: fetch failed {url}: {e}", file=sys.stderr)
        return None


def _fetch_url_bytes(
    session: FedHttpSession | None,
    url: str,
    *,
    offline: bool,
    warn: bool = True,
) -> bytes | None:
    if session is None or offline:
        return None
    try:
        return session.get_bytes(url)
    except Exception:
        if warn:
            print(f"WARN: fetch failed {url}", file=sys.stderr)
        return None


def monetary_cache_path_for_date(cache_dir: Path, ymd: str) -> Path | None:
    cand = cache_dir / f"_newsevents_pressreleases_monetary{ymd}a.htm.html"
    if cand.is_file():
        return cand
    hits = sorted(cache_dir.glob(f"_newsevents_pressreleases_monetary{ymd}*.html"))
    return hits[0] if hits else None


def fetch_fomc_html_for_meeting(
    cache_dir: Path,
    session: FedHttpSession | None,
    ymd: str,
    *,
    offline: bool,
) -> str | None:
    p = monetary_cache_path_for_date(cache_dir, ymd)
    if p and p.is_file():
        return p.read_text(encoding="utf-8", errors="replace")
    last_err: Exception | None = None
    for url in meeting_statement_urls(ymd):
        try:
            if session is None or offline:
                break
            html = session.get_text(url)
            if len(extract_fomc_statement_body(html)) >= 80:
                return html
        except Exception as e:
            last_err = e
            continue
    if last_err:
        print(f"WARN: FOMC {ymd} no usable statement HTML ({last_err})", file=sys.stderr)
    return None


def _try_meeting_document(
    urls: list[str],
    *,
    session: FedHttpSession | None,
    offline: bool,
    text_parser: callable,
    min_len: int = 120,
) -> tuple[str, str, list[str]]:
    """Return ``(text, source_url, quality_flags)`` trying each URL in order.

    Flags:
      - ``no_text`` (+ either ``http_error`` or ``pdf_unreadable``) when nothing usable found
      - ``short_body`` when content returned but below ``min_len`` words
    The caller decides whether to store a short body or treat it as missing.
    """

    flags: list[str] = []
    http_error_seen = False
    pdf_unreadable_seen = False
    best_short: tuple[str, str] | None = None
    for url in urls:
        if url.lower().endswith(".pdf"):
            data = _fetch_url_bytes(session, url, offline=offline, warn=False)
            if not data:
                http_error_seen = True
                continue
            doc = extract_pdf_text(data)
            if not doc:
                pdf_unreadable_seen = True
                continue
        else:
            raw = _fetch_url_text(session, url, offline=offline, warn=False)
            if not raw:
                http_error_seen = True
                continue
            doc = text_parser(raw)
        if len(doc.split()) >= min_len:
            return doc, url, []
        if doc and best_short is None:
            best_short = (doc, url)
    if best_short is not None:
        return best_short[0], best_short[1], [FLAG_SHORT_BODY]
    if http_error_seen:
        flags.append(FLAG_HTTP_ERROR)
    if pdf_unreadable_seen:
        flags.append(FLAG_PDF_UNREADABLE)
    flags.append(FLAG_NO_TEXT)
    return "", urls[0] if urls else "", flags


def build_fomc_dataframe(
    *,
    cache_dir: Path,
    calendar_path: Path | None,
    min_ymd: str,
    max_ymd: str,
    session: FedHttpSession | None,
    fred_bounds: pd.DataFrame | None,
    offline: bool,
    extra_calendar_html: str | None = None,
) -> pd.DataFrame:
    meetings = all_meeting_dates(
        calendar_path,
        min_ymd=min_ymd,
        max_ymd=max_ymd,
        extra_calendar_html=extra_calendar_html,
    )
    rows: list[dict[str, Any]] = []
    last_labels = {"decision": "maintain", "high": "0", "low": "0"}
    for ymd in meetings:
        html = fetch_fomc_html_for_meeting(cache_dir, session, ymd, offline=offline)
        stmt_url = meeting_statement_urls(ymd)[0]
        doc = ""
        stmt_flags: list[str] = []
        decision, high, low = last_labels["decision"], last_labels["high"], last_labels["low"]
        labels_from_fred = False
        if html:
            doc = extract_fomc_statement_body(html)
            if not doc.strip():
                stmt_flags.append(FLAG_NO_TEXT)
            pol = parse_monetary_statement(html)
            if "error" not in pol:
                decision = str(pol["decision"])
                high = str(pol["high"])
                low = str(pol["low"])
                last_labels = {"decision": decision, "high": high, "low": low}
            elif fred_bounds is not None and not offline:
                fb = infer_policy_from_fred(fred_bounds, ymd)
                decision, high, low = str(fb["decision"]), str(fb["high"]), str(fb["low"])
                last_labels = {"decision": decision, "high": high, "low": low}
                labels_from_fred = True
        else:
            stmt_flags.append(FLAG_NO_TEXT)
            if fred_bounds is not None and not offline:
                fb = infer_policy_from_fred(fred_bounds, ymd)
                decision, high, low = str(fb["decision"]), str(fb["high"]), str(fb["low"])
                last_labels = {"decision": decision, "high": high, "low": low}
                labels_from_fred = True

        rows.append(
            {
                "date": ymd,
                "type": "statement",
                "decision": decision,
                "high": high,
                "low": low,
                "document": doc,
                "word_count": int(len(doc.split())),
                "source_url": stmt_url,
                "quality_flags": list(stmt_flags),
                "labels_from_fred": labels_from_fred,
                "parser_version": PARSER_VERSION,
            }
        )

        minutes_doc, minutes_url, minutes_flags = _try_meeting_document(
            meeting_minutes_urls(ymd),
            session=session,
            offline=offline,
            text_parser=extract_minutes_body,
            min_len=250,
        )
        if minutes_doc or minutes_flags:
            rows.append(
                {
                    "date": ymd,
                    "type": "minutes",
                    "decision": decision,
                    "high": high,
                    "low": low,
                    "document": minutes_doc,
                    "word_count": int(len(minutes_doc.split())),
                    "source_url": minutes_url,
                    "quality_flags": list(minutes_flags),
                    "labels_from_fred": False,
                    "parser_version": PARSER_VERSION,
                }
            )

        press_doc, press_url, press_flags = _try_meeting_document(
            meeting_press_conference_urls(ymd),
            session=session,
            offline=offline,
            text_parser=lambda text: text,
            min_len=250,
        )
        if press_doc or press_flags:
            rows.append(
                {
                    "date": ymd,
                    "type": "press-conference",
                    "decision": decision,
                    "high": high,
                    "low": low,
                    "document": press_doc,
                    "word_count": int(len(press_doc.split())),
                    "source_url": press_url,
                    "quality_flags": list(press_flags),
                    "labels_from_fred": False,
                    "parser_version": PARSER_VERSION,
                }
            )

    df = pd.DataFrame(rows)
    # Keep the extended columns; callers that want the legacy pickle schema
    # should project to ``FOMC_COLUMNS_LEGACY`` explicitly.
    base_cols = list(FOMC_COLUMNS_LEGACY)
    extras = [c for c in ("source_url", "quality_flags", "labels_from_fred", "parser_version") if c in df.columns]
    return df[base_cols + extras]


def load_speeches_json_metadata(json_cache_path: Path) -> dict[str, str]:
    if not json_cache_path.is_file():
        return {}
    raw = json_cache_path.read_text(encoding="utf-8", errors="replace")
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    data = json.loads(raw)
    out: dict[str, str] = {}
    for row in data:
        link = str(row.get("l") or "")
        if not link.startswith("/newsevents/speech/"):
            continue
        slug = Path(link).stem
        name = str(row.get("s") or "").strip()
        if slug:
            out[slug] = name
    return out


def load_speeches_json_rows(
    json_cache_path: Path,
    session: FedHttpSession | None,
    *,
    offline: bool,
) -> list[dict[str, Any]]:
    raw: str | None = None
    if json_cache_path.is_file():
        raw = json_cache_path.read_text(encoding="utf-8", errors="replace")
    elif session is not None and not offline:
        try:
            raw = session.get_text(SPEECHES_JSON)
        except Exception as e:
            print(f"WARN: could not fetch {SPEECHES_JSON}: {e}", file=sys.stderr)
    if not raw:
        return []
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    return json.loads(raw)


def _discover_board_event_urls(
    *,
    min_year: int,
    max_year: int,
    session: FedHttpSession | None,
    offline: bool,
) -> set[str]:
    urls: set[str] = set()
    for kind in ("speech", "testimony"):
        for year in range(min_year, max_year + 1):
            raw = _fetch_url_text(session, board_year_listing_url(kind, year), offline=offline, warn=False)
            if not raw:
                continue
            hits = re.findall(rf'href="(/newsevents/{kind}/[a-z0-9]+\.htm)"', raw, flags=re.I)
            urls.update(BASE + href for href in hits)
    return urls


def _record_from_board_page(
    *,
    html: str,
    url: str,
    fallback_participant: str = "",
    fallback_title: str = "",
) -> dict[str, Any] | None:
    parsed = parse_board_event_page(html)
    slug = Path(urlparse(url).path).stem
    ymd = parsed["date_ymd"]
    if not ymd:
        m = _YMD_IN_SLUG_RE.search(slug)
        ymd = m.group(1) if m else ""
    participant = parsed["participant"] or fallback_participant
    title = parsed["title"] or fallback_title
    doc_body = parsed["document"]
    doc = f"{title}. {doc_body}".strip() if title else doc_body
    if not participant or not ymd or len(doc) < 40:
        return None
    wc = int(len(doc.split()))
    flags: list[str] = [FLAG_SHORT_BODY] if wc < 120 else []
    return {
        "date": ymd,
        "domain": "www.federalreserve.gov",
        "participant": participant,
        "document": doc,
        "word_count": wc,
        "source_url": url,
        "quality_flags": flags,
        "parser_version": PARSER_VERSION,
        "_slug": slug,
    }


def build_speaker_dataframe(
    *,
    cache_dir: Path,
    json_cache_path: Path,
    min_ymd: str,
    max_ymd: str,
    session: FedHttpSession | None = None,
    offline: bool = True,
) -> pd.DataFrame:
    rows_json = load_speeches_json_rows(json_cache_path, session, offline=offline)
    meta: dict[str, str] = dict(load_speeches_json_metadata(json_cache_path))
    records: dict[str, dict[str, Any]] = {}

    for row in rows_json:
        link = str(row.get("l") or "")
        if not link.startswith("/newsevents/speech/"):
            continue
        slug = Path(link).stem
        title = str(row.get("t") or "").strip()
        meta[slug] = str(row.get("s") or "").strip() or meta.get(slug, "")
        try:
            ymd = speech_date_to_ymd(str(row.get("d") or ""))
        except ValueError:
            continue
        if not (min_ymd <= ymd <= max_ymd):
            continue
        url = BASE + link
        raw = _fetch_url_text(session, url, offline=offline)
        if not raw:
            cache_p = cache_dir / f"{_cache_key_from_url(url)}.html"
            if not cache_p.is_file():
                continue
            raw = cache_p.read_text(encoding="utf-8", errors="replace")
        rec = _record_from_board_page(
            html=raw,
            url=url,
            fallback_participant=meta.get(slug, ""),
            fallback_title=title,
        )
        if rec:
            records[slug] = rec

    min_year = int(min_ymd[:4])
    max_year = int(max_ymd[:4])
    for url in sorted(_discover_board_event_urls(min_year=min_year, max_year=max_year, session=session, offline=offline)):
        m = _BOARD_DOC_RE.search(urlparse(url).path)
        if not m:
            continue
        slug = m.group(2)
        if slug in records:
            continue
        raw = _fetch_url_text(session, url, offline=offline)
        if not raw:
            cache_p = cache_dir / f"{_cache_key_from_url(url)}.html"
            if not cache_p.is_file():
                continue
            raw = cache_p.read_text(encoding="utf-8", errors="replace")
        rec = _record_from_board_page(
            html=raw,
            url=url,
            fallback_participant=meta.get(slug, ""),
        )
        if not rec:
            continue
        if not (min_ymd <= str(rec["date"]) <= max_ymd):
            continue
        records[slug] = rec

    for p in sorted(cache_dir.glob("_newsevents_speech_*.html")):
        m = _CACHE_SPEECH_RE.search(p.name)
        if not m:
            continue
        slug = f"{m.group(1)}{m.group(2)}{m.group(3)}"
        if slug in records:
            continue
        ymd = m.group(2)
        if not (min_ymd <= ymd <= max_ymd):
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        rec = _record_from_board_page(
            html=raw,
            url=f"{BASE}/newsevents/speech/{slug}.htm",
            fallback_participant=meta.get(slug, m.group(1).title()),
        )
        if rec:
            records[slug] = rec

    if not records:
        return pd.DataFrame(columns=[*SPEAKER_COLUMNS])
    sp = pd.DataFrame(list(records.values()))
    sp = sp.sort_values(["date", "participant"], kind="mergesort").reset_index(drop=True)
    return sp


# FOMC blackout: second Saturday before the meeting through the day after
# (see https://www.federalreserve.gov/monetarypolicy/mpr_communications.htm).
# Approximation used here: 12 days before through 1 day after.
BLACKOUT_PRE_DAYS = 12
BLACKOUT_POST_DAYS = 1

ALIGN_BLACKOUT_PRE = "blackout_pre"
ALIGN_POST_MEETING = "post_meeting"
ALIGN_NO_PRIOR_MEETING = "no_prior_meeting"


def _align_speech_to_meeting(
    sp_ts: pd.Timestamp,
    meetings: pd.DataFrame,
) -> tuple[str, str, str, str, str]:
    """Return ``(fomc_ref_date, decision, high, low, alignment_rule)``.

    Rules:
      - If ``sp_ts`` falls inside the blackout window of meeting N, map to N
        (pre-meeting speeches reflect the context for that decision).
      - Otherwise, map to the most recent **prior** meeting (the speech
        reflects the current stance set at that meeting). This replaces the
        old "nearest in time" rule, which mislabeled post-meeting speeches
        with the *next* decision they could not yet know about.
      - If no prior meeting exists, fall back to the first meeting and mark
        the row with ``no_prior_meeting`` for auditability.
    """

    for _, row in meetings.iterrows():
        mtg = row["_mtg"]
        start = mtg - pd.Timedelta(days=BLACKOUT_PRE_DAYS)
        end = mtg + pd.Timedelta(days=BLACKOUT_POST_DAYS)
        if start <= sp_ts <= end:
            return (
                str(row["date"]),
                str(row["decision"]),
                str(row["high"]),
                str(row["low"]),
                ALIGN_BLACKOUT_PRE,
            )
    prior = meetings[meetings["_mtg"] <= sp_ts]
    if not prior.empty:
        row = prior.iloc[-1]
        return (
            str(row["date"]),
            str(row["decision"]),
            str(row["high"]),
            str(row["low"]),
            ALIGN_POST_MEETING,
        )
    row = meetings.iloc[0]
    return (
        str(row["date"]),
        str(row["decision"]),
        str(row["high"]),
        str(row["low"]),
        ALIGN_NO_PRIOR_MEETING,
    )


def merge_speakers_nearest_fomc(speaker: pd.DataFrame, fomc: pd.DataFrame) -> pd.DataFrame:
    """Attach FOMC decision labels to each speech using blackout-aware rules.

    Historical behavior labelled each speech with the *temporally nearest*
    meeting, which leaks future information for speeches given right before
    a meeting and mislabels post-meeting speeches. The new logic classifies
    each speech into a pre-meeting *blackout_pre* bucket (maps to that
    meeting) or a *post_meeting* bucket (maps to the most recent prior
    meeting), and records the rule that fired in an ``alignment_rule``
    column for downstream auditing.
    """

    speaker_cols_ext = list(SPEAKER_COLUMNS) + ["alignment_rule"]
    if speaker.empty:
        return pd.DataFrame(columns=speaker_cols_ext)
    if fomc.empty:
        out = speaker.drop(columns=["_slug"], errors="ignore").copy()
        out["fomc-ref-date"] = ""
        out["decision"] = "maintain"
        out["high"] = "0"
        out["low"] = "0"
        out["alignment_rule"] = ALIGN_NO_PRIOR_MEETING
        keep = [c for c in speaker_cols_ext if c in out.columns]
        extras = [c for c in ("source_url", "quality_flags", "parser_version") if c in out.columns]
        return out[keep + extras]

    meetings = fomc.copy()
    meetings["_mtg"] = pd.to_datetime(meetings["date"].astype(str), format="%Y%m%d", errors="coerce")
    meetings = meetings.dropna(subset=["_mtg"]).sort_values("_mtg").reset_index(drop=True)
    meetings = meetings[["_mtg", "date", "decision", "high", "low"]]

    s = speaker.copy()
    s["_sp"] = pd.to_datetime(s["date"].astype(str), format="%Y%m%d", errors="coerce")
    s = s.dropna(subset=["_sp"]).reset_index(drop=True)

    refs: list[str] = []
    decs: list[str] = []
    hs: list[str] = []
    ls: list[str] = []
    rules: list[str] = []
    for sp_ts in s["_sp"]:
        ref, dec, hi, lo, rule = _align_speech_to_meeting(sp_ts, meetings)
        refs.append(ref)
        decs.append(dec)
        hs.append(hi)
        ls.append(lo)
        rules.append(rule)

    out = s.drop(columns=["_sp", "_slug"], errors="ignore").copy()
    out["fomc-ref-date"] = refs
    out["decision"] = decs
    out["high"] = hs
    out["low"] = ls
    out["alignment_rule"] = rules
    keep = [c for c in speaker_cols_ext if c in out.columns]
    extras = [c for c in ("source_url", "quality_flags", "parser_version") if c in out.columns]
    return out[keep + extras]


def validate_frames(fomc: pd.DataFrame, speaker: pd.DataFrame) -> None:
    for c in FOMC_COLUMNS:
        if c not in fomc.columns:
            raise ValueError(f"fomc missing {c}")
    for c in SPEAKER_COLUMNS:
        if c not in speaker.columns:
            raise ValueError(f"speaker missing {c}")
