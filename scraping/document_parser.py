"""Robust document parser for federalreserve.gov content.

Handles post-2020 edge cases where the Board site uses alternate press-release
letter suffixes (``monetary{ymd}a.htm`` vs ``b``/``c``), layout variants, and
the increasing reliance on PDFs for minutes and press conferences.

Design goals:

- Never silently produce a "stub" row. If extraction fails, return a
  :class:`ParsedDocument` with ``text=""`` and a populated ``quality_flags``
  list so the caller can decide how to store / retry.
- Prefer PyMuPDF (``fitz``) for PDF text. Fall back to ``pypdf`` only if
  PyMuPDF is unavailable at runtime.
- HTML parsing tries multiple Fed layout containers (``div#article``,
  ``div.col-xs-12.col-sm-8.col-md-8``, ``main``, ``article``, ``body``).
"""

from __future__ import annotations

import io
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

from bs4 import BeautifulSoup

try:
    import fitz  # type: ignore[import-untyped]

    _HAS_PYMUPDF = True
except Exception:  # pragma: no cover - PyMuPDF optional fallback path
    _HAS_PYMUPDF = False

try:
    from pypdf import PdfReader

    _HAS_PYPDF = True
except Exception:  # pragma: no cover
    _HAS_PYPDF = False


PARSER_VERSION = "2026.04.17"

# Quality flag vocabulary (keep stable; written to the audit table).
FLAG_NO_TEXT = "no_text"
FLAG_PDF_UNREADABLE = "pdf_unreadable"
FLAG_HTML_STRUCTURE_UNKNOWN = "html_structure_unknown"
FLAG_HTTP_404 = "http_404"
FLAG_HTTP_ERROR = "http_error"
FLAG_SHORT_BODY = "short_body"
FLAG_PARTIAL = "partial"


_ARTICLE_SELECTORS = (
    'div[id="article"]',
    "div.col-xs-12.col-sm-8.col-md-8",
    "article",
    "main",
    '[role="main"]',
    "body",
)

_BOILERPLATE_LINES = {
    "Share",
    "Watch Live",
    "Last Update:",
    "Back to Top",
    "Home",
}

# Unicode dashes/hyphens to normalize.
_DASH_CHARS = ("\u2011", "\u2010", "\u2012", "\u2013", "\u2014", "\u2212")
_WS = re.compile(r"\s+")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:Page\s+)?\d{1,3}\s*$", re.I)
_LETTER_SUFFIXES: tuple[str, ...] = ("a", "b", "c")

_BASE = "https://www.federalreserve.gov"


@dataclass
class ParsedDocument:
    """Result of a parse attempt.

    ``text`` is the final cleaned body. An empty string combined with one or
    more ``quality_flags`` means extraction failed and the row should be
    stored with that metadata rather than a fabricated placeholder.
    """

    text: str
    source_url: str
    doc_type: str
    quality_flags: list[str] = field(default_factory=list)
    word_count: int = 0

    def __post_init__(self) -> None:
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())
        if not self.text and FLAG_NO_TEXT not in self.quality_flags:
            self.quality_flags.append(FLAG_NO_TEXT)


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    for ch in _DASH_CHARS:
        t = t.replace(ch, "-")
    # Join hyphenated words across line breaks ("pol-\nicy" -> "policy").
    t = _HYPHEN_LINEBREAK.sub(r"\1\2", t)
    # Drop obvious page numbers and boilerplate lines.
    kept: list[str] = []
    for line in t.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _PAGE_NUMBER_LINE.match(stripped):
            continue
        if stripped in _BOILERPLATE_LINES:
            continue
        kept.append(stripped)
    joined = " ".join(kept)
    return _WS.sub(" ", joined).strip()


def _pick_main_node(soup: BeautifulSoup) -> Optional[object]:
    for sel in _ARTICLE_SELECTORS:
        node = soup.select_one(sel)
        if node is not None:
            return node
    return None


def _html_to_text(html: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    if not html:
        return "", [FLAG_NO_TEXT]
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "form", "noscript"]):
        tag.decompose()
    node = _pick_main_node(soup)
    if node is None:
        flags.append(FLAG_HTML_STRUCTURE_UNKNOWN)
        return "", flags
    raw = node.get_text("\n", strip=True)
    text = _normalize(raw)
    if not text:
        flags.append(FLAG_HTML_STRUCTURE_UNKNOWN)
    return text, flags


def _pdf_to_text_pymupdf(data: bytes) -> str:
    pages: list[str] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            try:
                pages.append(page.get_text("text") or "")
            except Exception:  # pragma: no cover - per-page robustness
                continue
    return "\n".join(pages)


def _pdf_to_text_pypdf(data: bytes) -> str:
    if not _HAS_PYPDF:
        return ""
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
    return "\n".join(pages)


def _pdf_to_text(data: bytes) -> tuple[str, list[str]]:
    flags: list[str] = []
    if not data:
        return "", [FLAG_NO_TEXT]
    raw = ""
    if _HAS_PYMUPDF:
        try:
            raw = _pdf_to_text_pymupdf(data)
        except Exception as exc:  # pragma: no cover
            print(f"WARN: PyMuPDF extract failed, falling back: {exc}", file=sys.stderr)
            raw = ""
    if not raw:
        raw = _pdf_to_text_pypdf(data)
    text = _normalize(raw)
    if not text:
        flags.append(FLAG_PDF_UNREADABLE)
    return text, flags


def meeting_statement_candidate_urls(ymd: str) -> list[str]:
    """All press-release URL variants we try for a given meeting date.

    The Fed uses ``monetary{ymd}a.htm`` for the statement and sometimes a
    ``b`` or ``c`` letter for an implementation note or late-day addendum.
    """

    return [f"{_BASE}/newsevents/pressreleases/monetary{ymd}{letter}.htm" for letter in _LETTER_SUFFIXES]


def meeting_minutes_candidate_urls(ymd: str) -> list[str]:
    return [
        f"{_BASE}/monetarypolicy/fomcminutes{ymd}.htm",
        f"{_BASE}/monetarypolicy/files/fomcminutes{ymd}.pdf",
        f"https://fraser.stlouisfed.org/files/text/historical/FOMC/meetingdocuments/fomcminutes{ymd}.txt",
    ]


def meeting_press_conference_candidate_urls(ymd: str) -> list[str]:
    return [
        f"{_BASE}/mediacenter/files/FOMCpresconf{ymd}.pdf",
        f"{_BASE}/mediacenter/files/FOMCpresconf{ymd}.pdf?stream=top",
        f"https://fraser.stlouisfed.org/files/text/historical/FOMC/meetingdocuments/FOMCpresconf{ymd}_final.txt",
    ]


TextFetcher = Callable[[str], Optional[str]]
BytesFetcher = Callable[[str], Optional[bytes]]


class DocumentParser:
    """High-level parser that picks the best candidate URL.

    Callers provide two tiny fetcher callables so the parser stays free of
    network dependencies and is easy to unit-test with in-memory stubs.

    Typical usage inside :mod:`rebuild.assemble`::

        parser = DocumentParser()
        doc = parser.parse_statement(ymd, fetch_text=session.try_get_text)
        if doc.text:
            ...
        else:
            # store row with doc.quality_flags for later --force-refresh-stubs
            ...
    """

    def __init__(self, *, min_statement_words: int = 80, min_long_form_words: int = 250) -> None:
        self.min_statement_words = min_statement_words
        self.min_long_form_words = min_long_form_words
        self.parser_version = PARSER_VERSION

    def parse_statement_html(self, html: str, *, source_url: str) -> ParsedDocument:
        text, flags = _html_to_text(html)
        wc = len(text.split()) if text else 0
        if text and wc < self.min_statement_words:
            flags.append(FLAG_SHORT_BODY)
        return ParsedDocument(
            text=text,
            source_url=source_url,
            doc_type="statement",
            quality_flags=flags,
            word_count=wc,
        )

    def parse_minutes(self, data: str | bytes, *, source_url: str) -> ParsedDocument:
        if isinstance(data, bytes):
            text, flags = _pdf_to_text(data)
        else:
            text, flags = _html_to_text(data)
        wc = len(text.split()) if text else 0
        if text and wc < self.min_long_form_words:
            flags.append(FLAG_SHORT_BODY)
        return ParsedDocument(
            text=text,
            source_url=source_url,
            doc_type="minutes",
            quality_flags=flags,
            word_count=wc,
        )

    def parse_press_conference(self, data: str | bytes, *, source_url: str) -> ParsedDocument:
        if isinstance(data, bytes):
            text, flags = _pdf_to_text(data)
        else:
            # Fraser serves transcripts as plain text.
            text = _normalize(data)
            flags = [] if text else [FLAG_NO_TEXT]
        wc = len(text.split()) if text else 0
        if text and wc < self.min_long_form_words:
            flags.append(FLAG_SHORT_BODY)
        return ParsedDocument(
            text=text,
            source_url=source_url,
            doc_type="press-conference",
            quality_flags=flags,
            word_count=wc,
        )

    def parse_speech(self, html: str, *, source_url: str) -> ParsedDocument:
        text, flags = _html_to_text(html)
        wc = len(text.split()) if text else 0
        if text and wc < self.min_statement_words:
            flags.append(FLAG_SHORT_BODY)
        return ParsedDocument(
            text=text,
            source_url=source_url,
            doc_type="speech",
            quality_flags=flags,
            word_count=wc,
        )

    def fetch_statement(self, ymd: str, *, fetch_text: TextFetcher) -> ParsedDocument:
        """Try statement URL variants (``a``/``b``/``c``), keep the longest body."""
        best: ParsedDocument | None = None
        last_flags: list[str] = []
        for url in meeting_statement_candidate_urls(ymd):
            html = fetch_text(url)
            if html is None:
                last_flags = [FLAG_HTTP_ERROR]
                continue
            candidate = self.parse_statement_html(html, source_url=url)
            if candidate.text and (best is None or candidate.word_count > best.word_count):
                best = candidate
        if best is not None:
            return best
        return ParsedDocument(
            text="",
            source_url=meeting_statement_candidate_urls(ymd)[0],
            doc_type="statement",
            quality_flags=last_flags or [FLAG_NO_TEXT],
        )

    def fetch_minutes(
        self,
        ymd: str,
        *,
        fetch_text: TextFetcher,
        fetch_bytes: BytesFetcher,
    ) -> ParsedDocument:
        return self._fetch_long_form(
            urls=meeting_minutes_candidate_urls(ymd),
            doc_type="minutes",
            fetch_text=fetch_text,
            fetch_bytes=fetch_bytes,
        )

    def fetch_press_conference(
        self,
        ymd: str,
        *,
        fetch_text: TextFetcher,
        fetch_bytes: BytesFetcher,
    ) -> ParsedDocument:
        return self._fetch_long_form(
            urls=meeting_press_conference_candidate_urls(ymd),
            doc_type="press-conference",
            fetch_text=fetch_text,
            fetch_bytes=fetch_bytes,
        )

    def _fetch_long_form(
        self,
        *,
        urls: list[str],
        doc_type: str,
        fetch_text: TextFetcher,
        fetch_bytes: BytesFetcher,
    ) -> ParsedDocument:
        accumulated_flags: list[str] = []
        for url in urls:
            if url.lower().endswith(".pdf") or "stream=top" in url.lower():
                data = fetch_bytes(url)
                if not data:
                    accumulated_flags.append(FLAG_HTTP_ERROR)
                    continue
                parser_fn = self.parse_minutes if doc_type == "minutes" else self.parse_press_conference
                candidate = parser_fn(data, source_url=url)
            else:
                raw = fetch_text(url)
                if raw is None:
                    accumulated_flags.append(FLAG_HTTP_ERROR)
                    continue
                if doc_type == "minutes":
                    candidate = self.parse_minutes(raw, source_url=url)
                else:
                    candidate = self.parse_press_conference(raw, source_url=url)
            if candidate.text and candidate.word_count >= self.min_long_form_words:
                return candidate
            if candidate.text:
                # Keep as partial candidate — return only if nothing better found.
                if FLAG_PARTIAL not in candidate.quality_flags:
                    candidate.quality_flags.append(FLAG_PARTIAL)
                return candidate
        return ParsedDocument(
            text="",
            source_url=urls[0],
            doc_type=doc_type,
            quality_flags=accumulated_flags or [FLAG_NO_TEXT],
        )


__all__ = [
    "DocumentParser",
    "ParsedDocument",
    "PARSER_VERSION",
    "FLAG_NO_TEXT",
    "FLAG_PDF_UNREADABLE",
    "FLAG_HTML_STRUCTURE_UNKNOWN",
    "FLAG_HTTP_404",
    "FLAG_HTTP_ERROR",
    "FLAG_SHORT_BODY",
    "FLAG_PARTIAL",
    "meeting_statement_candidate_urls",
    "meeting_minutes_candidate_urls",
    "meeting_press_conference_candidate_urls",
]
