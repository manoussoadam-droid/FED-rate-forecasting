"""Tests for core.repository (schema, upsert, audit SQLite)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from core.repository import (
    DocumentRepository,
    compute_quality_score,
    content_hash,
    normalize_fomc_frame,
    normalize_speaker_frame,
)


def _fomc_input() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "20230322",
                "type": "statement",
                "decision": "raise",
                "high": "4.75",
                "low": "5",
                "document": "Good body text " * 50,
                "word_count": 100,
                "source_url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230322a.htm",
                "quality_flags": [],
                "labels_from_fred": False,
                "parser_version": "test",
            },
            {
                "date": "20230503",
                "type": "minutes",
                "decision": "raise",
                "high": "5",
                "low": "5.25",
                "document": "",
                "word_count": 0,
                "source_url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes20230503.htm",
                "quality_flags": ["no_text", "pdf_unreadable"],
                "labels_from_fred": False,
                "parser_version": "test",
            },
        ]
    )


def _speaker_input() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fomc-ref-date": "20230322",
                "date": "20230315",
                "decision": "raise",
                "high": "4.75",
                "low": "5",
                "domain": "www.federalreserve.gov",
                "participant": "Jerome Powell",
                "document": "Speech body " * 60,
                "word_count": 120,
                "source_url": "https://www.federalreserve.gov/newsevents/speech/powell20230315a.htm",
                "quality_flags": [],
                "parser_version": "test",
                "alignment_rule": "blackout_pre",
            }
        ]
    )


def test_content_hash_is_stable_and_whitespace_insensitive() -> None:
    a = content_hash("  hello world  ")
    b = content_hash("hello world")
    assert a == b
    assert len(a) == 64


def test_compute_quality_score_bounds() -> None:
    assert compute_quality_score(800, []) == 1.0
    assert 0.0 <= compute_quality_score(10, ["no_text", "pdf_unreadable", "http_error"]) <= 1.0


def test_normalize_fomc_frame_schema() -> None:
    df = normalize_fomc_frame(_fomc_input())
    assert set(["date", "year", "document_type", "text_content", "content_hash", "quality_score", "quality_flags"]) <= set(df.columns)
    assert df["year"].tolist() == [2023, 2023]
    assert df["content_hash"].iloc[0] != df["content_hash"].iloc[1]


def test_repository_roundtrip_and_upsert(tmp_path: Path) -> None:
    repo = DocumentRepository(
        fomc_dir=tmp_path / "fomc",
        speaker_dir=tmp_path / "speaker",
        audit_db=tmp_path / "audit.sqlite",
    )
    repo.write_fomc(_fomc_input())
    repo.write_speaker(_speaker_input())

    fomc = repo.read_fomc()
    assert len(fomc) == 2
    speaker = repo.read_speaker()
    assert len(speaker) == 1

    # Re-writing the same frame is idempotent (content_hash dedupe).
    repo.write_fomc(_fomc_input())
    assert len(repo.read_fomc()) == 2

    # Stub URLs list picks up the minutes row with no_text / pdf_unreadable.
    stubs = repo.stub_urls()
    assert any("fomcminutes20230503" in u for u, _ in stubs)


def test_repository_audit_tables_populated(tmp_path: Path) -> None:
    repo = DocumentRepository(
        fomc_dir=tmp_path / "fomc",
        speaker_dir=tmp_path / "speaker",
        audit_db=tmp_path / "audit.sqlite",
    )
    run_id = repo.start_run()
    repo.write_fomc(_fomc_input())
    repo.log_fetch_error(run_id, "https://example.com/404", 404, "missing")
    repo.finish_run(run_id, rows_fomc=2, rows_speaker=0)

    with sqlite3.connect(tmp_path / "audit.sqlite") as cx:
        runs = cx.execute("SELECT rows_fomc, rows_speaker FROM ingestion_log WHERE run_id=?", (run_id,)).fetchone()
        assert runs == (2, 0)
        errs = cx.execute("SELECT COUNT(*) FROM fetch_errors WHERE run_id=?", (run_id,)).fetchone()[0]
        assert errs == 1
        audited = cx.execute("SELECT COUNT(*) FROM document_audit").fetchone()[0]
        assert audited == 2



def test_normalize_speaker_frame_preserves_fomc_ref_date() -> None:
    df = normalize_speaker_frame(_speaker_input())
    assert df["fomc_ref_date"].iloc[0] == "20230322"
    assert df["alignment_rule"].iloc[0] == "blackout_pre"
