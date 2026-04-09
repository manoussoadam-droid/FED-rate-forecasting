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
) -> str:
    for url in urls:
        if url.lower().endswith(".pdf"):
            data = _fetch_url_bytes(session, url, offline=offline, warn=False)
            if not data:
                continue
            doc = extract_pdf_text(data)
        else:
            raw = _fetch_url_text(session, url, offline=offline, warn=False)
            if not raw:
                continue
            doc = text_parser(raw)
        if len(doc.split()) >= min_len:
            return doc
    return ""


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
        doc = ""
        decision, high, low = last_labels["decision"], last_labels["high"], last_labels["low"]
        if html:
            doc = extract_fomc_statement_body(html)
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
                if not doc.strip():
                    doc = (
                        f"FOMC statement for {ymd}: text unavailable or unparseable ({pol.get('error')}); "
                        f"labels from FRED DFEDTARL/DFEDTARU."
                    )
            elif not doc.strip():
                doc = f"FOMC statement for {ymd}: parse error ({pol.get('error')})."
        elif fred_bounds is not None and not offline:
            fb = infer_policy_from_fred(fred_bounds, ymd)
            decision, high, low = str(fb["decision"]), str(fb["high"]), str(fb["low"])
            last_labels = {"decision": decision, "high": high, "low": low}
            doc = (
                f"FOMC statement for {ymd}: HTML unavailable after URL attempts; labels from FRED. "
                f"Check federalreserve.gov press release path for this date."
            )
        else:
            doc = (
                f"FOMC placeholder {ymd}: no HTML in cache; labels forward-filled from last "
                f"parsed meeting (offline mode)."
            )

        rows.append(
            {
                "date": ymd,
                "type": "statement",
                "decision": decision,
                "high": high,
                "low": low,
                "document": doc,
                "word_count": int(len(doc.split())),
            }
        )

        minutes_doc = _try_meeting_document(
            meeting_minutes_urls(ymd),
            session=session,
            offline=offline,
            text_parser=extract_minutes_body,
            min_len=250,
        )
        if minutes_doc:
            rows.append(
                {
                    "date": ymd,
                    "type": "minutes",
                    "decision": decision,
                    "high": high,
                    "low": low,
                    "document": minutes_doc,
                    "word_count": int(len(minutes_doc.split())),
                }
            )

        press_doc = _try_meeting_document(
            meeting_press_conference_urls(ymd),
            session=session,
            offline=offline,
            text_parser=lambda text: text,
            min_len=250,
        )
        if press_doc:
            rows.append(
                {
                    "date": ymd,
                    "type": "press-conference",
                    "decision": decision,
                    "high": high,
                    "low": low,
                    "document": press_doc,
                    "word_count": int(len(press_doc.split())),
                }
            )

    df = pd.DataFrame(rows)
    return df[FOMC_COLUMNS_LEGACY]


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
    return {
        "date": ymd,
        "domain": "www.federalreserve.gov",
        "participant": participant,
        "document": doc,
        "word_count": int(len(doc.split())),
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


def merge_speakers_nearest_fomc(speaker: pd.DataFrame, fomc: pd.DataFrame) -> pd.DataFrame:
    """Label speeches with merge_asof (backward + forward) and pick the temporally nearest FOMC meeting."""
    if speaker.empty:
        return pd.DataFrame(columns=[*SPEAKER_COLUMNS])
    if fomc.empty:
        out = speaker.drop(columns=["_slug"], errors="ignore").copy()
        out["fomc-ref-date"] = ""
        out["decision"] = "maintain"
        out["high"] = "0"
        out["low"] = "0"
        return out[SPEAKER_COLUMNS]

    fomc2 = fomc.copy()
    fomc2["_mtg"] = pd.to_datetime(fomc2["date"].astype(str), format="%Y%m%d", errors="coerce")
    fomc2 = fomc2.dropna(subset=["_mtg"]).sort_values("_mtg")
    # Only join keys + labels — avoid clashing document/word_count with speech columns.
    fomc_join = fomc2[["_mtg", "date", "decision", "high", "low"]]

    s = speaker.copy()
    s["_sp"] = pd.to_datetime(s["date"].astype(str), format="%Y%m%d", errors="coerce")
    s = s.dropna(subset=["_sp"]).sort_values("_sp")

    back = fomc_join.rename(
        columns={
            "date": "fomc_ref_b",
            "decision": "d_b",
            "high": "h_b",
            "low": "l_b",
            "_mtg": "mtg_b",
        }
    )
    fwd = fomc_join.rename(
        columns={
            "date": "fomc_ref_f",
            "decision": "d_f",
            "high": "h_f",
            "low": "l_f",
            "_mtg": "mtg_f",
        }
    )

    bw = pd.merge_asof(s, back, left_on="_sp", right_on="mtg_b", direction="backward")
    fw = pd.merge_asof(s, fwd, left_on="_sp", right_on="mtg_f", direction="forward")

    refs: list[str] = []
    decs: list[str] = []
    hs: list[str] = []
    ls: list[str] = []
    for i in range(len(bw)):
        sp_i = bw.iloc[i]["_sp"]
        tb, tf = bw.iloc[i]["mtg_b"], fw.iloc[i]["mtg_f"]
        dist_b = abs((sp_i - tb).total_seconds()) if pd.notna(tb) else float("inf")
        dist_f = abs((sp_i - tf).total_seconds()) if pd.notna(tf) else float("inf")
        if dist_b <= dist_f:
            refs.append(str(bw.iloc[i]["fomc_ref_b"]))
            decs.append(str(bw.iloc[i]["d_b"]))
            hs.append(str(bw.iloc[i]["h_b"]))
            ls.append(str(bw.iloc[i]["l_b"]))
        else:
            refs.append(str(fw.iloc[i]["fomc_ref_f"]))
            decs.append(str(fw.iloc[i]["d_f"]))
            hs.append(str(fw.iloc[i]["h_f"]))
            ls.append(str(fw.iloc[i]["l_f"]))

    out = bw.drop(
        columns=["mtg_b", "fomc_ref_b", "d_b", "h_b", "l_b", "_sp"],
        errors="ignore",
    ).copy()
    out["fomc-ref-date"] = refs
    out["decision"] = decs
    out["high"] = hs
    out["low"] = ls
    out = out.drop(columns=["_slug"], errors="ignore")
    return out[SPEAKER_COLUMNS]


def validate_frames(fomc: pd.DataFrame, speaker: pd.DataFrame) -> None:
    for c in FOMC_COLUMNS:
        if c not in fomc.columns:
            raise ValueError(f"fomc missing {c}")
    for c in SPEAKER_COLUMNS:
        if c not in speaker.columns:
            raise ValueError(f"speaker missing {c}")
