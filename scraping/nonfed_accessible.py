"""Direct-URL scrapers for currently accessible non-Fed media domains.

This module is intentionally conservative:
- it only supports domains we verified as fetchable from this environment
- it extracts from direct article URLs, not site search/index pages
- it does not try to defeat paywalls or anti-bot protections
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

_WS = re.compile(r"\s+")
_ALL_CAPS_PROMO = re.compile(r"^[A-Z0-9 '’\":,().!?\-]{8,}$")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class ExtractedArticle:
    url: str
    domain: str
    title: str
    date_ymd: str
    text: str


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def _parse_date(date_str: str) -> str:
    s = (date_str or "").strip()
    if not s:
        return ""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
    ):
        try:
            return datetime.strptime(s, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    s = re.sub(r"\s+\d{1,2}:\d{2}[ap]m\s+[A-Z]{2,4}$", "", s, flags=re.I)
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return ""


def _fetch_html(url: str, timeout: float = 45.0) -> str:
    r = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def _domain(url: str) -> str:
    return (urlparse(url).netloc or "").lower().replace("m.", "").replace("amp.", "")


def _title_from_meta(soup: BeautifulSoup) -> str:
    for attr in (("property", "og:title"), ("name", "title")):
        tag = soup.find("meta", attrs={attr[0]: attr[1]})
        if tag and tag.get("content"):
            return _norm(tag["content"])
    if soup.title:
        return _norm(soup.title.get_text(" ", strip=True))
    return ""


def _filter_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        t = _norm(line)
        if not t:
            continue
        if len(t) < 4:
            continue
        if _ALL_CAPS_PROMO.fullmatch(t):
            continue
        if t.startswith("GET ") or t.startswith("Watch ") or t.startswith("Read more"):
            continue
        if "CLICK HERE" in t:
            continue
        if re.search(r"\((?:Bloomberg|Getty Images|iStock|Reuters|AP)\)", t):
            continue
        out.append(t)
    return out


def _extract_cnbc(soup: BeautifulSoup, url: str) -> ExtractedArticle:
    body = soup.select_one("div.ArticleBody-articleBody")
    if not body:
        raise ValueError("CNBC body not found")
    lines = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    text = _norm("\n".join(_filter_lines(lines)))
    if len(text) < 300:
        raise ValueError("CNBC extracted text too short")
    meta_date = soup.find("meta", attrs={"property": "article:published_time"})
    date_ymd = _parse_date(meta_date.get("content", "") if meta_date else "")
    return ExtractedArticle(
        url=url,
        domain=_domain(url),
        title=_title_from_meta(soup),
        date_ymd=date_ymd,
        text=text,
    )


def _extract_foxbusiness(soup: BeautifulSoup, url: str) -> ExtractedArticle:
    body = soup.find("article") or soup.find("main")
    if not body:
        raise ValueError("Fox body not found")
    lines = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    text = _norm("\n".join(_filter_lines(lines)))
    if len(text) < 300:
        raise ValueError("Fox extracted text too short")
    meta_date = soup.find("meta", attrs={"name": "dc.date"})
    if meta_date and meta_date.get("content"):
        date_ymd = _parse_date(meta_date["content"])
    else:
        tnode = soup.find("time")
        date_ymd = _parse_date(tnode.get_text(" ", strip=True) if tnode else "")
    return ExtractedArticle(
        url=url,
        domain=_domain(url),
        title=_title_from_meta(soup),
        date_ymd=date_ymd,
        text=text,
    )


def fetch_article(url: str) -> ExtractedArticle:
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "lxml")
    dom = _domain(url)
    if "cnbc.com" in dom:
        return _extract_cnbc(soup, url)
    if "foxbusiness.com" in dom:
        return _extract_foxbusiness(soup, url)
    raise ValueError(f"unsupported or currently unreliable domain: {dom}")
