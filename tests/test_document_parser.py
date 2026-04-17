"""Tests for the new DocumentParser (HTML + candidate URL selection)."""

from __future__ import annotations

import pytest

from scraping.document_parser import (
    FLAG_HTML_STRUCTURE_UNKNOWN,
    FLAG_NO_TEXT,
    FLAG_SHORT_BODY,
    DocumentParser,
    meeting_minutes_candidate_urls,
    meeting_press_conference_candidate_urls,
    meeting_statement_candidate_urls,
)


POLICY_HTML = """
<html><body>
  <nav>Site nav</nav>
  <div id="article">
    <h3>Press Release</h3>
    <p>The Committee decided to maintain the target range for the federal funds
    rate at 5-1/4 to 5-1/2 percent.</p>
    <p>{filler}</p>
    <p>Voting for the monetary policy action were: Jerome H. Powell, Chair ...</p>
  </div>
  <div>Share</div>
</body></html>
"""


ALT_LAYOUT_HTML = """
<html><body>
  <div class="col-xs-12 col-sm-8 col-md-8">
    <p>Alternate layout body. {filler}</p>
  </div>
</body></html>
"""


def _filler(words: int = 200) -> str:
    return ("The Committee seeks to achieve maximum employment and inflation. " * (words // 10 + 1))


def test_parse_statement_finds_article_div() -> None:
    html = POLICY_HTML.format(filler=_filler(300))
    parser = DocumentParser()
    doc = parser.parse_statement_html(html, source_url="https://example/monetary20230322a.htm")
    assert doc.text
    assert doc.word_count > 80
    assert FLAG_NO_TEXT not in doc.quality_flags


def test_parse_statement_flags_short_body() -> None:
    html = POLICY_HTML.format(filler="only a few more words here")
    parser = DocumentParser()
    doc = parser.parse_statement_html(html, source_url="https://example/monetary20230322a.htm")
    assert doc.text
    assert FLAG_SHORT_BODY in doc.quality_flags


def test_parse_statement_flags_missing_structure() -> None:
    parser = DocumentParser()
    doc = parser.parse_statement_html("<html><body></body></html>", source_url="https://example/")
    assert doc.text == ""
    assert FLAG_NO_TEXT in doc.quality_flags
    assert FLAG_HTML_STRUCTURE_UNKNOWN in doc.quality_flags


def test_alternate_layout_picked_up() -> None:
    html = ALT_LAYOUT_HTML.format(filler=_filler(300))
    parser = DocumentParser()
    doc = parser.parse_statement_html(html, source_url="https://example/monetary20230322a.htm")
    assert doc.text


def test_fetch_statement_prefers_longer_variant() -> None:
    parser = DocumentParser()
    a_html = POLICY_HTML.format(filler="short body only")
    b_html = POLICY_HTML.format(filler=_filler(400))

    def fetch_text(url: str) -> str | None:
        if url.endswith("monetary20230322a.htm"):
            return a_html
        if url.endswith("monetary20230322b.htm"):
            return b_html
        return None

    doc = parser.fetch_statement("20230322", fetch_text=fetch_text)
    assert doc.text
    assert "monetary20230322b.htm" in doc.source_url


def test_candidate_urls_cover_expected_variants() -> None:
    stmt = meeting_statement_candidate_urls("20230322")
    assert any(u.endswith("monetary20230322a.htm") for u in stmt)
    assert any(u.endswith("monetary20230322b.htm") for u in stmt)
    mins = meeting_minutes_candidate_urls("20230322")
    assert any(u.endswith(".pdf") for u in mins)
    pcs = meeting_press_conference_candidate_urls("20230322")
    assert any("FOMCpresconf20230322" in u for u in pcs)


def test_parser_never_fabricates_text_on_404() -> None:
    parser = DocumentParser()

    def fetch_text(url: str) -> str | None:
        return None

    def fetch_bytes(url: str) -> bytes | None:
        return None

    stmt = parser.fetch_statement("20230322", fetch_text=fetch_text)
    assert stmt.text == ""
    assert FLAG_NO_TEXT in stmt.quality_flags

    pc = parser.fetch_press_conference(
        "20230322", fetch_text=fetch_text, fetch_bytes=fetch_bytes
    )
    assert pc.text == ""
    assert FLAG_NO_TEXT in pc.quality_flags
