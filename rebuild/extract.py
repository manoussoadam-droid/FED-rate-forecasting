"""BeautifulSoup extraction + regex cleanup for Fed HTML bodies."""

from __future__ import annotations

import io
import re
from datetime import datetime

from bs4 import BeautifulSoup
from pypdf import PdfReader

from rebuild.clean import normalize_document_text
from scraping.fed_official import normalize_fed_text


_DATE_LINE_RE = re.compile(
    r"^(?:\d{1,2}/\d{1,2}/\d{4}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4})$"
)
_PARTICIPANT_LINE_RE = re.compile(
    r"^(?:Chair|Vice Chair|Vice Chair for Supervision|Chair Pro Tempore|Governor|Member|Director)\b"
)
_CONTEXT_LINE_RE = re.compile(r"^(?:At|Before)\b")
_NUMERIC_FOOTNOTE_RE = re.compile(r"^\d+$")


def _main_node_from_html(html: str) -> BeautifulSoup | None:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    return soup.select_one('div[id="article"]') or soup.select_one("article") or soup.body


def _clean_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        t = normalize_document_text(normalize_fed_text(line))
        if not t:
            continue
        if t in {"Share", "Watch Live"}:
            continue
        if _NUMERIC_FOOTNOTE_RE.fullmatch(t):
            continue
        out.append(t)
    return out


def extract_fomc_statement_body(html: str) -> str:
    main = _main_node_from_html(html)
    if not main:
        return ""
    text = main.get_text("\n", strip=True)
    return normalize_document_text(normalize_fed_text(text))


def extract_speech_body(html: str) -> str:
    main = _main_node_from_html(html)
    if not main:
        return ""
    text = main.get_text("\n", strip=True)
    return normalize_document_text(normalize_fed_text(text))


def extract_minutes_body(html: str) -> str:
    return extract_speech_body(html)


def extract_pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception:
        return ""
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return normalize_document_text("\n".join(pages))


def parse_board_event_page(html: str) -> dict[str, str]:
    main = _main_node_from_html(html)
    if not main:
        return {"date_ymd": "", "title": "", "participant": "", "document": ""}
    lines = _clean_lines(main.get_text("\n", strip=True))
    if not lines:
        return {"date_ymd": "", "title": "", "participant": "", "document": ""}

    date_ymd = ""
    idx = 0
    if _DATE_LINE_RE.fullmatch(lines[0]):
        for fmt in ("%B %d, %Y", "%m/%d/%Y"):
            try:
                date_ymd = datetime.strptime(lines[0], fmt).strftime("%Y%m%d")
                break
            except ValueError:
                continue
        idx = 1

    title = lines[idx] if idx < len(lines) else ""
    participant = ""
    body_start = idx + 1
    if body_start < len(lines) and (
        _PARTICIPANT_LINE_RE.search(lines[body_start])
        or (
            body_start + 1 < len(lines)
            and not _CONTEXT_LINE_RE.search(lines[body_start])
            and _CONTEXT_LINE_RE.search(lines[body_start + 1])
        )
    ):
        participant = lines[body_start]
        body_start += 1
    if body_start < len(lines) and _CONTEXT_LINE_RE.search(lines[body_start]):
        body_start += 1
    while body_start < len(lines) and lines[body_start] in {"Share", "Watch Live"}:
        body_start += 1
    body = " ".join(lines[body_start:]).strip()
    return {
        "date_ymd": date_ymd,
        "title": title,
        "participant": participant,
        "document": body,
    }
