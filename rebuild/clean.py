"""Regex-heavy normalization after HTML extraction (syllabus-style cleaning)."""

from __future__ import annotations

import html as html_lib
import re
import unicodedata

# Residual tags / script fragments if any slipped through BeautifulSoup
_TAG_RE = re.compile(r"<[^>]+>")
_STYLE_BLOCK_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_WS = re.compile(r"\s+")
_NON_PRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def strip_residual_markup(text: str) -> str:
    t = _COMMENT_RE.sub(" ", text)
    t = _STYLE_BLOCK_RE.sub(" ", t)
    t = _TAG_RE.sub(" ", t)
    return t


def normalize_unicode_dashes(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    for ch in ("\u2011", "\u2010", "\u2012", "\u2013", "\u2014", "\u2212"):
        t = t.replace(ch, "-")
    return t


def normalize_document_text(text: str) -> str:
    """Collapse whitespace, unescape entities, strip markup noise."""
    t = html_lib.unescape(text or "")
    t = strip_residual_markup(t)
    t = normalize_unicode_dashes(t)
    t = _NON_PRINTABLE.sub(" ", t)
    return _WS.sub(" ", t).strip()
