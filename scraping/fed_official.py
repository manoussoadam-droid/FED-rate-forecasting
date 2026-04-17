"""Fetch structured content from federalreserve.gov (speeches index, FOMC statements).

Respect robots.txt and site terms; use a descriptive User-Agent and modest rate limits.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

SPEECHES_JSON = "https://www.federalreserve.gov/json/ne-speeches.json"
FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
SPEECHES_TESTIMONY_URL = "https://www.federalreserve.gov/newsevents/speeches-testimony.htm"
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
    """Tiny requests wrapper with an on-disk cache and refresh controls.

    The cache is intentionally simple (one file per URL path). A sidecar
    ``{key}.meta.json`` records when the file was fetched and the parser
    version that produced the last ingested row, which is used by the
    ``--force-refresh-stubs`` workflow to invalidate entries whose audited
    quality flags are still ``no_text`` / ``pdf_unreadable`` / similar.

    Parameters
    ----------
    force_refresh:
        If ``True``, every call bypasses the on-disk cache and refetches.
    refresh_if:
        Optional predicate ``(url) -> bool``. When it returns ``True`` the
        cached file is ignored for that URL (the network result overwrites
        the cache). This is how the rebuild script schedules targeted
        re-fetches of previously stubbed rows.
    """

    def __init__(
        self,
        cache_dir: Path | None,
        delay_s: float,
        *,
        force_refresh: bool = False,
        refresh_if: Callable[[str], bool] | None = None,
        refresh_urls: Iterable[str] | None = None,
    ) -> None:
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": DEFAULT_UA})
        self.cache_dir = cache_dir
        self.delay_s = delay_s
        self.force_refresh = force_refresh
        self.refresh_if = refresh_if
        self._refresh_urls: set[str] = set(refresh_urls or [])
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _should_refresh(self, url: str) -> bool:
        if self.force_refresh:
            return True
        if url in self._refresh_urls:
            return True
        if self.refresh_if is not None:
            try:
                return bool(self.refresh_if(url))
            except Exception:
                return False
        return False

    def _write_sidecar(self, key: str, url: str) -> None:
        if not self.cache_dir:
            return
        meta_path = self.cache_dir / f"{key}.meta.json"
        try:
            meta_path.write_text(
                json.dumps(
                    {
                        "url": url,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def get_text(self, url: str) -> str:
        key = re.sub(r"[^\w.-]+", "_", urlparse(url).path)[:200]
        p = self.cache_dir / f"{key}.html" if self.cache_dir else None
        if p is not None and p.exists() and not self._should_refresh(url):
            return p.read_text(encoding="utf-8", errors="replace")
        time.sleep(self.delay_s)
        r = self.s.get(url, timeout=90)
        r.raise_for_status()
        r.encoding = "utf-8"
        text = r.text
        if p is not None:
            p.write_text(text, encoding="utf-8")
            self._write_sidecar(key, url)
        return text

    def get_bytes(self, url: str) -> bytes:
        path = urlparse(url).path
        suffix = Path(path).suffix or ".bin"
        key = re.sub(r"[^\w.-]+", "_", path)[:200]
        p = self.cache_dir / f"{key}{suffix}" if self.cache_dir else None
        if p is not None and p.exists() and not self._should_refresh(url):
            return p.read_bytes()
        time.sleep(self.delay_s)
        r = self.s.get(url, timeout=90)
        r.raise_for_status()
        data = r.content
        if p is not None:
            p.write_bytes(data)
            self._write_sidecar(key, url)
        return data


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


def meeting_statement_urls(ymd: str) -> list[str]:
    """Primary FOMC statement URLs to try (some dates use implementation note / alternate letter)."""
    return [
        f"{BASE}/newsevents/pressreleases/monetary{ymd}a.htm",
        f"{BASE}/newsevents/pressreleases/monetary{ymd}b.htm",
    ]


def meeting_minutes_urls(ymd: str) -> list[str]:
    return [
        f"{BASE}/monetarypolicy/fomcminutes{ymd}.htm",
        f"{BASE}/monetarypolicy/files/fomcminutes{ymd}.pdf",
        f"https://fraser.stlouisfed.org/files/text/historical/FOMC/meetingdocuments/fomcminutes{ymd}.txt",
    ]


def meeting_press_conference_urls(ymd: str) -> list[str]:
    return [
        f"{BASE}/mediacenter/files/FOMCpresconf{ymd}.pdf",
        f"{BASE}/mediacenter/files/FOMCpresconf{ymd}.pdf?stream=top",
        f"https://fraser.stlouisfed.org/files/text/historical/FOMC/meetingdocuments/FOMCpresconf{ymd}_final.txt",
    ]


def board_year_listing_url(kind: str, year: int) -> str:
    if kind == "speech":
        return f"{BASE}/newsevents/speech/{year}-speeches.htm"
    if kind == "testimony":
        return f"{BASE}/newsevents/testimony/{year}-testimony.htm"
    raise ValueError(f"unsupported Board event kind: {kind}")


def fetch_meeting_policy(session: FedHttpSession, ymd: str) -> dict[str, Any]:
    url = meeting_statement_url(ymd)
    html = session.get_text(url)
    parsed = parse_monetary_statement(html)
    parsed["meeting_date"] = ymd
    parsed["source_url"] = url
    return parsed
