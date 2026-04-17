"""Tests for blackout-aware speaker-to-meeting alignment."""

from __future__ import annotations

import pandas as pd

from rebuild.assemble import (
    ALIGN_BLACKOUT_PRE,
    ALIGN_NO_PRIOR_MEETING,
    ALIGN_POST_MEETING,
    merge_speakers_nearest_fomc,
)


def _fomc() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "20230322", "decision": "raise", "high": "4.75", "low": "5"},
            {"date": "20230503", "decision": "raise", "high": "5", "low": "5.25"},
            {"date": "20230614", "decision": "maintain", "high": "5", "low": "5.25"},
        ]
    )


def _speaker_row(date: str, participant: str) -> dict:
    return {
        "fomc-ref-date": "",
        "date": date,
        "decision": "",
        "high": "",
        "low": "",
        "domain": "www.federalreserve.gov",
        "participant": participant,
        "document": "body " * 150,
        "word_count": 150,
    }


def test_speech_in_blackout_maps_to_upcoming_meeting() -> None:
    speakers = pd.DataFrame(
        [
            _speaker_row("20230315", "Powell"),
            _speaker_row("20230321", "Williams"),
        ]
    )
    out = merge_speakers_nearest_fomc(speakers, _fomc())
    assert all(out["fomc-ref-date"] == "20230322")
    assert all(out["alignment_rule"] == ALIGN_BLACKOUT_PRE)
    assert out["decision"].iloc[0] == "raise"


def test_speech_after_meeting_maps_to_prior_meeting() -> None:
    speakers = pd.DataFrame([_speaker_row("20230410", "Waller")])
    out = merge_speakers_nearest_fomc(speakers, _fomc())
    assert out["fomc-ref-date"].iloc[0] == "20230322"
    assert out["alignment_rule"].iloc[0] == ALIGN_POST_MEETING


def test_speech_before_any_meeting_is_flagged() -> None:
    speakers = pd.DataFrame([_speaker_row("20220101", "Jefferson")])
    out = merge_speakers_nearest_fomc(speakers, _fomc())
    assert out["alignment_rule"].iloc[0] == ALIGN_NO_PRIOR_MEETING


def test_empty_speaker_returns_empty_frame_with_rule_column() -> None:
    out = merge_speakers_nearest_fomc(pd.DataFrame(), _fomc())
    assert "alignment_rule" in out.columns
    assert out.empty
