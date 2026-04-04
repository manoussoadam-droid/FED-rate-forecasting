"""Fetch structured content from federalreserve.gov (speeches index, FOMC statements).

Respect robots.txt and site terms; use a descriptive User-Agent and modest rate limits.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

SPEECHES_JSON = "https://www.federalreserve.gov/json/ne-speeches.json"
FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BASE = "https://www.federalreserve.gov"

DEFAULT_UA = (
    "Mozilla/5.0 (compatible; fomc_tools/1.0; educational research; +https://www.federalreserve.gov)"
)

_MINUTES_RE = re.compile(r"fomcminutes(20\d{6})\.pdf", re.I)
# After "federal funds rate", official wording uses "at X to Y" or ", to X to Y" (cuts/hikes).
_RANGE_AFTER_FUNDS = re.compile(
    r"(?:at|to)\s+([\d\-/]+)\s+to\s+([\d\-/]+)\s+percent",
    re.I,
)
_DECISION_RE = re.compile(
    r"to\s+(maintain|lower|raise)\s+the\s+target\s+range",
    re.I,
)


def rate_token_to_float(tok: str) -> float:
    tok = tok.strip()
    m = re.match(r"^(\d+)$", tok)
    if m:
        return float(int(m.group(1)))
    m = re.match(r"^(\d+)-(\d+)/(\d+)$", tok)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    m = re.match(r"^(\d+)/(\d+)$", tok)
    if m:
        return int(m.group(1)) / int(m.group(2))
    return float(tok)


def fmt_rate(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


class FedHttpSession:
    def __init__(self, cache_dir: Path | None, delay_s: float) -> None:
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": DEFAULT_UA})
        self.cache_dir = cache_dir
        self.delay_s = delay_s
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_text(self, url: str) -> str:
        key = re.sub(r"[^\w.-]+", "_", urlparse(url).path)[:200]
        if self.cache_dir:
            p = self.cache_dir / f"{key}.html"
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
        time.sleep(self.delay_s)
        r = self.s.get(url, timeout=90)
        r.raise_for_status()
        # federalreserve.gov pages are UTF-8; apparent_encoding can mis-decode narrow hyphens.
        r.encoding = "utf-8"
        text = r.text
        if self.cache_dir:
            p = self.cache_dir / f"{key}.html"
            p.write_text(text, encoding="utf-8")
        return text


def fetch_speeches_metadata(session: FedHttpSession) -> list[dict[str, Any]]:
    raw = session.get_text(SPEECHES_JSON)
    # Fed JSON is UTF-8 with BOM
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    return json.loads(raw)


def speech_date_to_ymd(d_field: str) -> str:
    """JSON 'd' like '3/31/2026 5:10:00 PM' -> YYYYMMDD string."""
    part = (d_field or "").strip().split()[0]
    dt = datetime.strptime(part, "%m/%d/%Y")
    return dt.strftime("%Y%m%d")


def extract_speech_body(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one('div[id="article"]') or soup.select_one("article") or soup.body
    if not main:
        return ""
    text = main.get_text("\n", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def normalize_fed_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    for ch in ("\u2011", "\u2010", "\u2012", "\u2013", "\u2014", "\u2212"):
        t = t.replace(ch, "-")
    return t


def parse_monetary_statement(html: str) -> dict[str, Any]:
    """Return decision + high/low (Fed convention in this repo: high=lower bound, low=upper bound)."""
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one('div[id="article"]') or soup.body
    if not main:
        return {"error": "no article body"}
    text = normalize_fed_text(main.get_text(" ", strip=True))
    i = text.lower().find("federal funds rate")
    if i < 0:
        return {"error": "no federal funds rate mention"}
    sub = text[i : i + 550]
    dm = _DECISION_RE.search(text)
    decision = dm.group(1).lower() if dm else "maintain"
    rm = _RANGE_AFTER_FUNDS.search(sub)
    if not rm:
        return {"error": "no target range", "decision": decision, "snippet": sub[:220]}
    lo = rate_token_to_float(rm.group(1))
    hi = rate_token_to_float(rm.group(2))
    return {"decision": decision, "high": fmt_rate(lo), "low": fmt_rate(hi)}


def scrape_minutes_meeting_dates(session: FedHttpSession) -> set[str]:
    html = session.get_text(FOMC_CALENDAR_URL)
    return set(_MINUTES_RE.findall(html))


def meeting_statement_url(ymd: str) -> str:
    return f"{BASE}/newsevents/pressreleases/monetary{ymd}a.htm"


def fetch_meeting_policy(session: FedHttpSession, ymd: str) -> dict[str, Any]:
    url = meeting_statement_url(ymd)
    html = session.get_text(url)
    parsed = parse_monetary_statement(html)
    parsed["meeting_date"] = ymd
    parsed["source_url"] = url
    return parsed
