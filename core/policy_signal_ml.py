"""Policy-signal ML utilities for early speech inference.

The product question is intentionally harder than plain document
classification: as a speech unfolds, decide whether it is rate-relevant and,
if so, whether it points to lower/maintain/raise.  This module keeps the
feature engineering, weak relevance labels, online-prefix evaluation, and
plotting in one reusable place so Streamlit and scripts share the same logic.
"""

from __future__ import annotations

import math
import re
import sqlite3
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from core.config import ARTIFACTS_DIR, AUDIT_DB
from core.ingest import load_fomc, load_speaker, parse_yyyymmdd
from core.text_clean import clean_for_ml

DIRECTION_LABELS = ("lower", "maintain", "raise")
ACTION_LABELS = ("no_rate_signal", "lower", "maintain", "raise", "uncertain_rate_signal")
FRED_CONTEXT_COLUMNS = (
    "fred_fedfunds",
    "fred_t10y2y",
    "fred_unrate",
    "fred_cpi",
    "fred_target_lower",
    "fred_target_upper",
    "fred_target_mid",
    "fred_fedfunds_delta_90d",
    "fred_fedfunds_delta_180d",
    "fred_target_mid_delta_90d",
    "fred_target_mid_delta_180d",
    "fred_t10y2y_delta_90d",
    "fred_unrate_delta_90d",
    "fred_cpi_yoy",
    "fred_context_available",
)
CLAUDE_FEATURE_COLUMNS = (
    "claude_rate_relevance",
    "claude_direction_lower",
    "claude_direction_maintain",
    "claude_direction_raise",
    "claude_confidence",
    "claude_hawkish_score",
    "claude_dovish_score",
    "claude_has_explicit_action",
)
STRUCTURED_CONTEXT_COLUMNS = (
    "target_lower",
    "target_upper",
    "target_mid",
    "target_spread",
    "alignment_days_to_ref",
    "alignment_pre_meeting",
    "alignment_abs_days_to_ref",
    "alignment_rule_blackout_pre",
    "alignment_rule_post_meeting",
    "alignment_rule_no_prior_meeting",
    "quality_score_feature",
    "labels_from_fred_feature",
)
CLAUDE_FEATURE_CACHE = ARTIFACTS_DIR / "policy_signal_tuning" / "claude_policy_features.csv"
CONTEXT_FEATURE_COLUMNS = FRED_CONTEXT_COLUMNS + STRUCTURED_CONTEXT_COLUMNS + CLAUDE_FEATURE_COLUMNS + ("speaker_tier",)
PRIMARY_ACTION_LABELS = ("no_rate_signal", "lower", "maintain", "raise")
MANUAL_FEATURE_VERSION = 3

POLICY_KEYWORDS = {
    "federal funds": 4.0,
    "target range": 4.0,
    "interest rate": 3.0,
    "interest rates": 3.0,
    "policy rate": 3.0,
    "monetary policy": 3.0,
    "fomc": 2.5,
    "inflation": 2.0,
    "price stability": 2.0,
    "maximum employment": 2.0,
    "labor market": 1.5,
    "unemployment": 1.5,
    "tightening": 1.5,
    "easing": 1.5,
    "rate hike": 2.0,
    "rate cut": 2.0,
    "balance sheet": 1.0,
    "outlook": 0.75,
}

HAWKISH_TERMS = (
    "inflation elevated",
    "inflation remains elevated",
    "inflation remains too high",
    "inflation is too high",
    "unacceptably high inflation",
    "persistent inflation",
    "restrictive",
    "restrictive stance",
    "restrictive policy",
    "keep policy restrictive",
    "sufficiently restrictive",
    "tightening",
    "further tightening",
    "additional policy firming",
    "policy firming",
    "raise rates",
    "rate hikes",
    "higher rates",
    "higher for longer",
    "price pressures",
    "upside risks to inflation",
    "tight labor market",
    "labor market remains strong",
)

DOVISH_TERMS = (
    "downside risks",
    "downside risks to employment",
    "weak demand",
    "recession",
    "rate cuts",
    "rate reductions",
    "lowering rates",
    "lower rates",
    "easing",
    "less restrictive",
    "reduce restriction",
    "easing restraint",
    "accommodative",
    "softening labor",
    "labor market has cooled",
    "cooling labor market",
    "progress on inflation",
    "confidence that inflation",
    "disinflation",
    "normalization of policy",
    "policy normalization",
    "gradual normalization",
)

HOLD_TERMS = (
    "data dependent",
    "incoming data",
    "proceed carefully",
    "careful approach",
    "wait and see",
    "well positioned",
    "balanced risks",
    "balance of risks",
    "hold rates",
    "keep rates",
    "maintain the target range",
    "target range unchanged",
    "remain restrictive",
)

UNCERTAINTY_TERMS = (
    "uncertainty",
    "uncertain",
    "risks",
    "headwinds",
    "mixed",
    "volatility",
    "geopolitical",
    "tariff",
    "tariffs",
)

DIRECT_POLICY_CUES = (
    "decided to maintain",
    "maintain the target range",
    "keep the target range",
    "leave the target range unchanged",
    "raise the target range",
    "increase the target range",
    "lower the target range",
    "decrease the target range",
    "cut interest rates",
    "raise interest rates",
    "lower interest rates",
    "hold rates",
)


@dataclass
class FeatureBundle:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD | None
    nmf: NMF | None
    kmeans: MiniBatchKMeans | None
    scaler: StandardScaler
    context_feature_names: tuple[str, ...] = ()
    manual_feature_version: int = MANUAL_FEATURE_VERSION


@dataclass
class PolicySignalArtifact:
    feature_bundle: FeatureBundle
    relevance_model: Any
    direction_model: Any
    thresholds: dict[str, float]
    metrics: dict[str, Any]
    feature_mode: str = "sparse"


def normalize_direction(value: Any) -> str:
    raw = str(value).strip().lower()
    mapping = {
        "fall": "lower",
        "lower": "lower",
        "cut": "lower",
        "decrease": "lower",
        "maintain": "maintain",
        "hold": "maintain",
        "unchanged": "maintain",
        "raise": "raise",
        "rise": "raise",
        "increase": "raise",
        "hike": "raise",
    }
    return mapping.get(raw, raw)


def policy_relevance_score(text: str) -> float:
    lower = str(text).lower()
    score = 0.0
    for term, weight in POLICY_KEYWORDS.items():
        score += weight * len(re.findall(re.escape(term), lower))
    return float(score)


def _term_hits(text: str, terms: tuple[str, ...]) -> int:
    lower = str(text).lower()
    return int(sum(len(re.findall(re.escape(term), lower)) for term in terms))


def _regex_hits(text: str, pattern: str) -> int:
    return int(len(re.findall(pattern, str(text).lower())))


def _numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _boolish_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    raw = df[column]
    if raw.dtype == bool:
        return raw.fillna(False)
    return raw.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def add_structured_policy_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add already-collected policy context fields for ML feature engineering.

    The source tables contain numeric rate bounds and FOMC reference dates. We
    normalize the potentially reversed high/low fields and derive blackout-style
    timing features from event date versus reference meeting date. These are
    model inputs only; they do not alter labels or corpus contents.
    """

    work = df.copy()
    raw_high = _numeric_column(work, "high", 0.0)
    raw_low = _numeric_column(work, "low", 0.0)
    work["target_lower"] = np.minimum(raw_high, raw_low)
    work["target_upper"] = np.maximum(raw_high, raw_low)
    work["target_mid"] = (work["target_lower"] + work["target_upper"]) / 2.0
    work["target_spread"] = (work["target_upper"] - work["target_lower"]).abs()

    ref = parse_yyyymmdd(work["fomc_ref_date"]) if "fomc_ref_date" in work.columns else work["event_date"]
    event = pd.to_datetime(work["event_date"], errors="coerce")
    days_to_ref = (pd.to_datetime(ref, errors="coerce") - event).dt.days
    work["alignment_days_to_ref"] = pd.to_numeric(days_to_ref, errors="coerce").fillna(0.0).astype(float)
    work["alignment_pre_meeting"] = work["alignment_days_to_ref"].between(0, 10).astype(float)
    work["alignment_abs_days_to_ref"] = work["alignment_days_to_ref"].abs().clip(upper=180).fillna(180.0)
    rule = work.get("alignment_rule", pd.Series([""] * len(work), index=work.index)).astype(str).str.lower()
    work["alignment_rule_blackout_pre"] = rule.eq("blackout_pre").astype(float)
    work["alignment_rule_post_meeting"] = rule.eq("post_meeting").astype(float)
    work["alignment_rule_no_prior_meeting"] = rule.eq("no_prior_meeting").astype(float)
    missing_rule = rule.eq("") | rule.eq("nan") | rule.eq("none")
    work.loc[missing_rule & work["alignment_pre_meeting"].eq(1.0), "alignment_rule_blackout_pre"] = 1.0
    work.loc[missing_rule & work["alignment_days_to_ref"].lt(0), "alignment_rule_post_meeting"] = 1.0

    quality = _numeric_column(work, "quality_score", 1.0).clip(lower=0.0, upper=1.0)
    work["quality_score_feature"] = quality
    work["labels_from_fred_feature"] = _boolish_column(work, "labels_from_fred").astype(float)
    return work


def document_feature_hash(row: pd.Series | dict[str, Any]) -> str:
    """Stable hash key for optional external feature caches."""

    getter = row.get if hasattr(row, "get") else lambda key, default="": default
    parts = [
        str(getter("source", "")),
        str(getter("event_date", ""))[:10],
        str(getter("speaker", "")),
        str(getter("domain", "")),
        str(getter("document", "")),
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()


def attach_claude_policy_features(
    df: pd.DataFrame,
    cache_path: Path = CLAUDE_FEATURE_CACHE,
) -> pd.DataFrame:
    """Attach cached Claude-derived labels/features if they have been built.

    This function intentionally never calls Claude. Networked labeling is done by
    a separate script and saved to CSV so model training remains deterministic.
    Missing cache rows are represented as zeros.
    """

    work = df.copy()
    if "doc_hash" not in work.columns:
        work["doc_hash"] = work.apply(document_feature_hash, axis=1)
    for col in CLAUDE_FEATURE_COLUMNS:
        work[col] = 0.0
    if not cache_path.is_file():
        return work
    try:
        cache = pd.read_csv(cache_path)
    except Exception:
        return work
    if "doc_hash" not in cache.columns:
        return work
    use_cols = ["doc_hash"] + [col for col in CLAUDE_FEATURE_COLUMNS if col in cache.columns]
    if len(use_cols) <= 1:
        return work
    cache = cache[use_cols].drop_duplicates("doc_hash", keep="last")
    merged = work.drop(columns=[col for col in CLAUDE_FEATURE_COLUMNS if col in work.columns]).merge(
        cache,
        on="doc_hash",
        how="left",
    )
    for col in CLAUDE_FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    return merged


def speaker_tier(value: Any, *, source: str = "", domain: str = "") -> float:
    """Map speaker identity into a small authority/market-attention feature.

    The exact number is deliberately simple and interpretable: official FOMC
    material and Chair speeches get the highest tier, Vice Chairs next,
    Governors next, and regional/other speakers lower. It is a helper feature,
    not a claim that lower-tier speeches never matter.
    """

    if str(source).lower() == "fomc":
        return 3.0
    text = f"{value} {domain}".lower()
    if "chair powell" in text or "jerome h. powell" in text or "jerome powell" in text:
        return 3.0
    if "vice chair" in text:
        return 2.0
    if "governor" in text or "board of governors" in text:
        return 1.5
    if "president" in text or "federal reserve bank" in text:
        return 1.0
    if "fomc" in text or "federal reserve" in text:
        return 0.75
    return 0.25


def load_fred_context(audit_db: Path = AUDIT_DB) -> pd.DataFrame:
    """Load cached FRED context from SQLite as a wide, date-indexed frame.

    The ML model uses the latest available observation at each document date.
    If the cache is absent, callers get an empty frame and context features
    safely fall back to zero with ``fred_context_available=0``.
    """

    if not audit_db.is_file():
        return pd.DataFrame()
    try:
        with sqlite3.connect(audit_db) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fred_cache'",
                conn,
            )
            if tables.empty:
                return pd.DataFrame()
            raw = pd.read_sql_query(
                """
                SELECT series_id, obs_date, value
                FROM fred_cache
                WHERE series_id IN ('FEDFUNDS', 'T10Y2Y', 'UNRATE', 'CPIAUCSL', 'DFEDTARL', 'DFEDTARU')
                """,
                conn,
            )
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()

    raw["obs_date"] = pd.to_datetime(raw["obs_date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["obs_date"])
    wide = (
        raw.pivot_table(index="obs_date", columns="series_id", values="value", aggfunc="last")
        .sort_index()
        .rename(
            columns={
                "FEDFUNDS": "fred_fedfunds",
                "T10Y2Y": "fred_t10y2y",
                "UNRATE": "fred_unrate",
                "CPIAUCSL": "fred_cpi",
                "DFEDTARL": "fred_target_lower",
                "DFEDTARU": "fred_target_upper",
            }
        )
    )
    base_cols = {
        "fred_fedfunds",
        "fred_t10y2y",
        "fred_unrate",
        "fred_cpi",
        "fred_target_lower",
        "fred_target_upper",
    }
    for col in base_cols:
        if col not in wide.columns:
            wide[col] = np.nan
    wide["fred_target_mid"] = wide[["fred_target_lower", "fred_target_upper"]].mean(axis=1)
    wide = wide.ffill()

    def _delta_days(frame: pd.DataFrame, col: str, days: int) -> pd.Series:
        current = frame[[col]].reset_index().rename(columns={"obs_date": "event_date", col: "current"})
        past = current.rename(columns={"current": "past"}).copy()
        past["event_date"] = past["event_date"] + pd.Timedelta(days=days)
        merged = pd.merge_asof(
            current.sort_values("event_date"),
            past.sort_values("event_date"),
            on="event_date",
            direction="backward",
        )
        return (merged["current"] - merged["past"]).reindex(current.index).to_numpy()

    wide["fred_fedfunds_delta_90d"] = _delta_days(wide, "fred_fedfunds", 90)
    wide["fred_fedfunds_delta_180d"] = _delta_days(wide, "fred_fedfunds", 180)
    wide["fred_target_mid_delta_90d"] = _delta_days(wide, "fred_target_mid", 90)
    wide["fred_target_mid_delta_180d"] = _delta_days(wide, "fred_target_mid", 180)
    wide["fred_t10y2y_delta_90d"] = _delta_days(wide, "fred_t10y2y", 90)
    wide["fred_unrate_delta_90d"] = _delta_days(wide, "fred_unrate", 90)
    wide["fred_cpi_yoy"] = _delta_days(wide, "fred_cpi", 365)
    wide["fred_context_available"] = wide[
        ["fred_fedfunds", "fred_t10y2y", "fred_unrate", "fred_target_mid"]
    ].notna().any(axis=1).astype(float)
    for col in FRED_CONTEXT_COLUMNS:
        if col not in wide.columns:
            wide[col] = np.nan
    return wide.reset_index().rename(columns={"obs_date": "event_date"})


def attach_fred_context(df: pd.DataFrame, fred_context: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach latest-prior FRED observations to each event date."""

    work = df.copy()
    if "event_date" not in work.columns:
        for col in FRED_CONTEXT_COLUMNS:
            work[col] = 0.0
        return work

    fred = load_fred_context() if fred_context is None else fred_context.copy()
    if fred.empty:
        for col in FRED_CONTEXT_COLUMNS:
            work[col] = 0.0
        return work

    left = work.reset_index().rename(columns={"index": "_row_order"})
    left["event_date"] = pd.to_datetime(left["event_date"], errors="coerce")
    left = left.sort_values("event_date")
    right = fred.copy()
    right["event_date"] = pd.to_datetime(right["event_date"], errors="coerce")
    right = right.dropna(subset=["event_date"]).sort_values("event_date")
    merged = pd.merge_asof(left, right, on="event_date", direction="backward")
    merged = merged.sort_values("_row_order").drop(columns=["_row_order"]).reset_index(drop=True)
    for col in FRED_CONTEXT_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged


def _context_values(context_rows: pd.DataFrame | None, n_rows: int, columns: tuple[str, ...]) -> np.ndarray:
    if not columns:
        return np.empty((n_rows, 0), dtype=float)
    if context_rows is None or len(context_rows) == 0:
        return np.zeros((n_rows, len(columns)), dtype=float)
    ctx = context_rows.reset_index(drop=True).copy()
    if len(ctx) == 1 and n_rows > 1:
        ctx = pd.concat([ctx] * n_rows, ignore_index=True)
    if len(ctx) != n_rows:
        ctx = ctx.reindex(range(n_rows))
    out = []
    for col in columns:
        if col in ctx.columns:
            out.append(pd.to_numeric(ctx[col], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        else:
            out.append(np.zeros(n_rows, dtype=float))
    return np.vstack(out).T


def _manual_feature_version_for_bundle(bundle: FeatureBundle) -> int:
    """Infer manual feature version from the fitted scaler when possible.

    Older joblib artifacts were saved before ``manual_feature_version`` existed.
    When those artifacts are unpickled with the current dataclass, Python may
    expose the new default value even though the scaler was fitted on the older
    seven-column manual block. The scaler shape is the source of truth.
    """

    expected = getattr(getattr(bundle, "scaler", None), "n_features_in_", None)
    if expected is None:
        return int(getattr(bundle, "manual_feature_version", 1))
    fixed_width = 0
    if getattr(bundle, "svd", None) is not None:
        fixed_width += int(getattr(bundle.svd, "n_components", 0))
    if getattr(bundle, "nmf", None) is not None:
        fixed_width += int(getattr(bundle.nmf, "n_components", 0))
    if getattr(bundle, "kmeans", None) is not None:
        fixed_width += int(getattr(bundle.kmeans, "n_clusters", 0))
    fixed_width += len(tuple(getattr(bundle, "context_feature_names", ()) or ()))
    manual_width = int(expected) - fixed_width
    if manual_width <= 7:
        return 1
    if manual_width <= 34:
        return 2
    return int(getattr(bundle, "manual_feature_version", MANUAL_FEATURE_VERSION))


def explicit_policy_cue_pct(text: str) -> float:
    """Return the first direct policy-action cue location as speech percent.

    If no direct phrase is present, returns NaN. This is a rough proxy for
    "before the speech explicitly says the rate action" and is used only as a
    speed diagnostic, not as a training feature.
    """

    lower = str(text).lower()
    first = None
    for phrase in DIRECT_POLICY_CUES:
        idx = lower.find(phrase)
        if idx >= 0:
            first = idx if first is None else min(first, idx)
    if first is None:
        return float("nan")
    before_words = len(lower[:first].split())
    total_words = max(1, len(lower.split()))
    return float(max(1, min(100, round(100 * before_words / total_words, 1))))


def make_policy_signal_frame(min_words: int = 80) -> pd.DataFrame:
    """Build the modeling table from current Parquet-backed corpora.

    `rate_relevant` is a weak label: FOMC documents are treated as policy
    relevant, while speeches/interviews must contain enough monetary-policy
    language to be treated as rate-relevant. Non-relevant speeches become the
    explicit `no_rate_signal` class, which is what the UI should show for
    speeches that are about supervision, banking plumbing, etc.
    """

    fomc = load_fomc().copy()
    speaker = load_speaker().copy()

    f = pd.DataFrame(
        {
            "source": "fomc",
            "event_date": parse_yyyymmdd(fomc["date"]),
            "speaker": fomc.get("speaker", pd.Series([""] * len(fomc), index=fomc.index)).astype(str),
            "domain": fomc.get("type", "fomc").astype(str),
            "document": fomc["document"].astype(str),
            "decision": fomc["decision"].map(normalize_direction),
            "high": fomc.get("high", 0),
            "low": fomc.get("low", 0),
            "fomc_ref_date": fomc["date"].astype(str),
            "word_count": fomc["word_count"].astype(int),
        }
    )
    for col in ("quality_score", "quality_flags", "labels_from_fred"):
        if col in fomc.columns:
            f[col] = fomc[col]
    s = pd.DataFrame(
        {
            "source": "speaker",
            "event_date": parse_yyyymmdd(speaker["date"]),
            "speaker": speaker["participant"].astype(str),
            "domain": speaker["domain"].astype(str),
            "document": speaker["document"].astype(str),
            "decision": speaker["decision"].map(normalize_direction),
            "high": speaker.get("high", 0),
            "low": speaker.get("low", 0),
            "fomc_ref_date": speaker.get("fomc-ref-date", speaker["date"]).astype(str),
            "word_count": speaker["word_count"].astype(int),
        }
    )
    for col in ("quality_score", "quality_flags", "labels_from_fred", "alignment_rule"):
        if col in speaker.columns:
            s[col] = speaker[col]
    df = pd.concat([f, s], ignore_index=True)
    df = add_structured_policy_context(df)
    df["doc_hash"] = df.apply(document_feature_hash, axis=1)
    df = attach_claude_policy_features(df)
    if "quality_score" in df.columns:
        quality = pd.to_numeric(df["quality_score"], errors="coerce").fillna(1.0)
        df = df[quality >= 0.4].copy()
        if "labels_from_fred" in df.columns:
            labels_from_fred = _boolish_column(df, "labels_from_fred")
            df = df[~(labels_from_fred & (quality.loc[df.index] < 0.7))].copy()
    if "quality_flags" in df.columns:
        bad_flags = {"no_text", "pdf_unreadable", "stub"}

        def _has_bad_flag(flags: Any) -> bool:
            if isinstance(flags, str):
                lower = flags.lower()
                return any(flag in lower for flag in bad_flags)
            if isinstance(flags, (list, tuple, set)):
                return bool(bad_flags.intersection({str(flag).lower() for flag in flags}))
            return False

        df = df[~df["quality_flags"].apply(_has_bad_flag)].copy()
    df = df[df["word_count"] >= int(min_words)].copy()
    df = df[df["decision"].isin(DIRECTION_LABELS)].copy()
    df = df.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)

    df["policy_score"] = df["document"].map(policy_relevance_score)
    df["policy_score_per_1k"] = df["policy_score"] / (df["word_count"].clip(lower=1) / 1000.0)
    df["hawkish_hits"] = df["document"].map(lambda t: _term_hits(t, HAWKISH_TERMS))
    df["dovish_hits"] = df["document"].map(lambda t: _term_hits(t, DOVISH_TERMS))
    df["hold_hits"] = df["document"].map(lambda t: _term_hits(t, HOLD_TERMS))
    df["uncertainty_hits"] = df["document"].map(lambda t: _term_hits(t, UNCERTAINTY_TERMS))
    df["speaker_tier"] = [
        speaker_tier(row.speaker, source=row.source, domain=row.domain)
        for row in df.itertuples(index=False)
    ]
    df = attach_fred_context(df)
    speaker_relevant = (df["policy_score"] >= 10) | (
        (df["policy_score"] >= 5) & (df["policy_score_per_1k"] >= 2.0)
    )
    df["rate_relevant"] = np.where(df["source"].eq("fomc") | speaker_relevant, 1, 0).astype(int)
    df["action_label"] = np.where(df["rate_relevant"].eq(1), df["decision"], "no_rate_signal")
    df["cue_pct"] = df["document"].map(explicit_policy_cue_pct)
    return df.reset_index(drop=True)


def weak_rate_relevance_for_text(text: str, source: str = "speaker") -> bool:
    """Apply the same weak relevance rule used to create training labels."""

    if str(source).lower() == "fomc":
        return True
    words = max(1, len(str(text).split()))
    score = policy_relevance_score(text)
    score_per_1k = score / (words / 1000.0)
    return bool((score >= 10) or ((score >= 5) and (score_per_1k >= 2.0)))


def weak_action_label(text: str, decision: Any, source: str = "speaker") -> str:
    if not weak_rate_relevance_for_text(text, source=source):
        return "no_rate_signal"
    return normalize_direction(decision)


def prefix_text(text: str, pct: int) -> str:
    words = str(text).split()
    if not words:
        return ""
    n = max(1, int(math.ceil(len(words) * max(1, min(100, pct)) / 100)))
    return " ".join(words[:n])


def expand_prefix_samples(df: pd.DataFrame, prefix_pcts: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        for pct in prefix_pcts:
            d = row._asdict()
            d["full_document"] = d["document"]
            d["document"] = prefix_text(str(row.document), int(pct))
            d["prefix_pct"] = int(pct)
            d["prefix_frac"] = float(pct) / 100.0
            rows.append(d)
    return pd.DataFrame(rows)


def fit_feature_bundle(
    texts: list[str],
    *,
    max_features: int,
    n_topics: int,
    svd_components: int,
    random_state: int,
    context_rows: pd.DataFrame | None = None,
) -> FeatureBundle:
    clean_texts = [clean_for_ml(t, 120_000) for t in texts]
    vectorizer = TfidfVectorizer(
        max_features=int(max_features),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english",
    )
    X = vectorizer.fit_transform(clean_texts)

    max_rank = max(2, min(X.shape[0] - 1, X.shape[1] - 1))
    svd_n = min(int(svd_components), max_rank)
    svd = TruncatedSVD(n_components=svd_n, random_state=random_state) if svd_n >= 2 else None
    nmf_n = min(int(n_topics), max_rank)
    nmf = NMF(n_components=nmf_n, init="nndsvda", max_iter=250, random_state=random_state) if nmf_n >= 2 else None
    kmeans = MiniBatchKMeans(n_clusters=min(max(2, nmf_n), max(2, X.shape[0])), random_state=random_state, n_init="auto")

    context_names = tuple(col for col in CONTEXT_FEATURE_COLUMNS if context_rows is not None and col in context_rows.columns)
    dense_parts = _dense_feature_parts(
        clean_texts,
        [1.0] * len(clean_texts),
        vectorizer,
        svd,
        nmf,
        kmeans,
        fit=True,
        context_rows=context_rows,
        context_feature_names=context_names,
    )
    scaler = StandardScaler()
    scaler.fit(dense_parts)
    return FeatureBundle(
        vectorizer=vectorizer,
        svd=svd,
        nmf=nmf,
        kmeans=kmeans,
        scaler=scaler,
        context_feature_names=context_names,
        manual_feature_version=MANUAL_FEATURE_VERSION,
    )


def _dense_feature_parts(
    clean_texts: list[str],
    prefix_fracs: list[float],
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD | None,
    nmf: NMF | None,
    kmeans: MiniBatchKMeans | None,
    *,
    fit: bool,
    context_rows: pd.DataFrame | None = None,
    context_feature_names: tuple[str, ...] = (),
    manual_feature_version: int = MANUAL_FEATURE_VERSION,
) -> np.ndarray:
    X = vectorizer.transform(clean_texts)
    parts: list[np.ndarray] = []
    if svd is not None:
        parts.append(svd.fit_transform(X) if fit else svd.transform(X))
    if nmf is not None:
        parts.append(nmf.fit_transform(X) if fit else nmf.transform(X))
    if kmeans is not None:
        if fit:
            clusters = kmeans.fit_predict(X)
        else:
            clusters = kmeans.predict(X)
        cluster_dense = np.zeros((len(clean_texts), int(kmeans.n_clusters)), dtype=float)
        cluster_dense[np.arange(len(clean_texts)), clusters] = 1.0
        parts.append(cluster_dense)

    context = _context_values(context_rows, len(clean_texts), context_feature_names)

    def _ctx(row_idx: int, name: str) -> float:
        if context.size == 0 or name not in context_feature_names:
            return 0.0
        return float(context[row_idx, context_feature_names.index(name)])

    manual = []
    for row_idx, (text, frac) in enumerate(zip(clean_texts, prefix_fracs)):
        wc = max(1, len(str(text).split()))
        per_1k = wc / 1000.0
        pscore = policy_relevance_score(text)
        hawk = _term_hits(text, HAWKISH_TERMS)
        dove = _term_hits(text, DOVISH_TERMS)
        base_features = [
            float(frac),
            math.log1p(wc),
            pscore,
            pscore / per_1k,
            float(hawk),
            float(dove),
            float(hawk - dove),
        ]
        if int(manual_feature_version) <= 1:
            manual.append(base_features)
            continue

        hold = _term_hits(text, HOLD_TERMS)
        uncertainty = _term_hits(text, UNCERTAINTY_TERMS)
        direct = _term_hits(text, DIRECT_POLICY_CUES)
        inflation_hits = _regex_hits(text, r"\binflation\b|\bprices?\b|\bcpi\b")
        labor_hits = _regex_hits(text, r"\blabor market\b|\bunemployment\b|\bemployment\b|\bpayrolls?\b")
        growth_hits = _regex_hits(text, r"\bgrowth\b|\bdemand\b|\bspending\b|\brecession\b|\bslowdown\b")

        hawk_rate = float(hawk) / per_1k
        dove_rate = float(dove) / per_1k
        hold_rate = float(hold) / per_1k
        uncertainty_rate = float(uncertainty) / per_1k
        tone_balance = (float(hawk) - float(dove)) / math.sqrt(wc)

        row_target_mid = _ctx(row_idx, "target_mid")
        fredfunds = _ctx(row_idx, "fred_fedfunds")
        fred_target_mid = _ctx(row_idx, "fred_target_mid")
        rate_level = row_target_mid or fredfunds or fred_target_mid
        target_spread = _ctx(row_idx, "target_spread")
        alignment_pre_meeting = _ctx(row_idx, "alignment_pre_meeting")
        alignment_abs_days = _ctx(row_idx, "alignment_abs_days_to_ref")
        alignment_rule_blackout = _ctx(row_idx, "alignment_rule_blackout_pre")
        alignment_rule_post = _ctx(row_idx, "alignment_rule_post_meeting")
        alignment_rule_none = _ctx(row_idx, "alignment_rule_no_prior_meeting")
        quality_score = _ctx(row_idx, "quality_score_feature")
        labels_from_fred = _ctx(row_idx, "labels_from_fred_feature")
        rate_high_gap = max(rate_level - 2.5, 0.0)
        rate_low_gap = max(2.5 - rate_level, 0.0)
        fedfunds_delta_180d = _ctx(row_idx, "fred_fedfunds_delta_180d")
        target_delta_180d = _ctx(row_idx, "fred_target_mid_delta_180d")
        unrate_delta = _ctx(row_idx, "fred_unrate_delta_90d")
        cpi_yoy = _ctx(row_idx, "fred_cpi_yoy")
        claude_relevance = _ctx(row_idx, "claude_rate_relevance")
        claude_lower = _ctx(row_idx, "claude_direction_lower")
        claude_maintain = _ctx(row_idx, "claude_direction_maintain")
        claude_raise = _ctx(row_idx, "claude_direction_raise")
        claude_hawkish = _ctx(row_idx, "claude_hawkish_score")
        claude_dovish = _ctx(row_idx, "claude_dovish_score")

        alignment_rule_features = (
            [
                alignment_rule_blackout,
                alignment_rule_post,
                alignment_rule_none,
            ]
            if int(manual_feature_version) >= 3
            else []
        )

        manual.append(
            base_features
            + [
                hawk_rate,
                dove_rate,
                hold_rate,
                uncertainty_rate,
                float(direct),
                float(inflation_hits) / per_1k,
                float(labor_hits) / per_1k,
                float(growth_hits) / per_1k,
                tone_balance,
                math.log1p(max(pscore / per_1k, 0.0)),
                rate_level * hawk_rate,
                rate_level * dove_rate,
                rate_high_gap * dove_rate,
                rate_low_gap * hawk_rate,
                abs(target_delta_180d or fedfunds_delta_180d) * hold_rate,
                unrate_delta * dove_rate,
                cpi_yoy * hawk_rate,
                target_spread,
                alignment_pre_meeting,
            ]
            + alignment_rule_features
            + [
                math.log1p(alignment_abs_days),
                quality_score,
                labels_from_fred,
                claude_relevance * pscore / per_1k,
                (claude_lower - claude_raise),
                claude_maintain * hold_rate,
                claude_hawkish * hawk_rate,
                claude_dovish * dove_rate,
            ]
        )
    parts.append(np.asarray(manual, dtype=float))
    if context.size:
        parts.append(context)
    return np.hstack(parts)


def transform_features(
    bundle: FeatureBundle,
    texts: list[str],
    prefix_fracs: list[float],
    context_rows: pd.DataFrame | None = None,
) -> sparse.csr_matrix:
    clean_texts = [clean_for_ml(t, 120_000) for t in texts]
    X_tfidf = bundle.vectorizer.transform(clean_texts)
    context_names = tuple(getattr(bundle, "context_feature_names", ()) or ())
    dense = _dense_feature_parts(
        clean_texts,
        prefix_fracs,
        bundle.vectorizer,
        bundle.svd,
        bundle.nmf,
        bundle.kmeans,
        fit=False,
        context_rows=context_rows,
        context_feature_names=context_names,
        manual_feature_version=_manual_feature_version_for_bundle(bundle),
    )
    dense_scaled = bundle.scaler.transform(dense)
    return sparse.hstack([X_tfidf, sparse.csr_matrix(dense_scaled)], format="csr")


def transform_dense_features(
    bundle: FeatureBundle,
    texts: list[str],
    prefix_fracs: list[float],
    context_rows: pd.DataFrame | None = None,
) -> np.ndarray:
    """Return the compact dense topic/manual feature block for neural models."""

    clean_texts = [clean_for_ml(t, 120_000) for t in texts]
    context_names = tuple(getattr(bundle, "context_feature_names", ()) or ())
    dense = _dense_feature_parts(
        clean_texts,
        prefix_fracs,
        bundle.vectorizer,
        bundle.svd,
        bundle.nmf,
        bundle.kmeans,
        fit=False,
        context_rows=context_rows,
        context_feature_names=context_names,
        manual_feature_version=_manual_feature_version_for_bundle(bundle),
    )
    return bundle.scaler.transform(dense)


def transform_for_artifact(
    artifact: PolicySignalArtifact,
    texts: list[str],
    prefix_fracs: list[float],
    context_rows: pd.DataFrame | None = None,
) -> Any:
    """Use the feature representation the saved model was trained with."""

    if str(getattr(artifact, "feature_mode", "sparse")).lower() == "dense":
        return transform_dense_features(artifact.feature_bundle, texts, prefix_fracs, context_rows=context_rows)
    return transform_features(artifact.feature_bundle, texts, prefix_fracs, context_rows=context_rows)


def probability_matrix(model: Any, X: Any, labels: list[Any]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        classes = list(getattr(model, "classes_", labels))
        out = np.zeros((X.shape[0], len(labels)), dtype=float)
        for idx, label in enumerate(labels):
            if label in classes:
                out[:, idx] = raw[:, classes.index(label)]
        row_sum = out.sum(axis=1, keepdims=True)
        return np.divide(out, row_sum, out=np.full_like(out, 1.0 / len(labels)), where=row_sum > 0)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        classes = list(getattr(model, "classes_", labels))
        if scores.ndim == 1:
            pos = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
            out = np.zeros((X.shape[0], len(labels)), dtype=float)
            if len(classes) == 2:
                for idx, label in enumerate(labels):
                    if label == classes[0]:
                        out[:, idx] = 1.0 - pos
                    elif label == classes[1]:
                        out[:, idx] = pos
            else:
                out[:, :] = 1.0 / len(labels)
            return out
        exps = np.exp(scores - scores.max(axis=1, keepdims=True))
        softmax = exps / exps.sum(axis=1, keepdims=True)
        out = np.zeros((X.shape[0], len(labels)), dtype=float)
        for idx, label in enumerate(labels):
            if label in classes:
                out[:, idx] = softmax[:, classes.index(label)]
        return out

    pred = model.predict(X)
    out = np.zeros((X.shape[0], len(labels)), dtype=float)
    for row_idx, value in enumerate(pred):
        if value in labels:
            out[row_idx, labels.index(value)] = 1.0
    return out


def classify_action(
    p_relevant: float,
    direction_probs: dict[str, float],
    *,
    relevance_threshold: float,
    no_signal_threshold: float,
    direction_threshold: float,
    margin_threshold: float,
    lower_override_threshold: float = 1.01,
    lower_override_gap: float = 0.0,
    threshold_family: str | None = None,
) -> dict[str, Any]:
    best_direction, best_prob = max(direction_probs.items(), key=lambda kv: kv[1])
    sorted_probs = sorted(direction_probs.values(), reverse=True)
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(best_prob)
    if p_relevant <= no_signal_threshold:
        return {"action": "no_rate_signal", "state": "no_rate_signal", "confidence": float(1 - p_relevant), "margin": margin}
    if p_relevant < relevance_threshold:
        return {"action": "uncertain_rate_signal", "state": "uncertain", "confidence": float(p_relevant), "margin": margin}
    lower_prob = float(direction_probs.get("lower", 0.0))
    maintain_prob = float(direction_probs.get("maintain", 0.0))
    if lower_prob >= float(lower_override_threshold) and (maintain_prob - lower_prob) <= float(lower_override_gap):
        return {
            "action": "lower",
            "state": "rate_relevant_lower_override",
            "confidence": lower_prob,
            "margin": margin,
        }
    if best_prob < direction_threshold or margin < margin_threshold:
        return {"action": "uncertain_rate_signal", "state": "rate_relevant_uncertain_direction", "confidence": float(best_prob), "margin": margin}
    return {"action": best_direction, "state": "rate_relevant_directional", "confidence": float(best_prob), "margin": margin}


def predict_action_frame(
    artifact: PolicySignalArtifact,
    df: pd.DataFrame,
    *,
    prefix_pct: int,
) -> pd.DataFrame:
    work = expand_prefix_samples(df, [int(prefix_pct)])
    X = transform_for_artifact(
        artifact,
        work["document"].astype(str).tolist(),
        work["prefix_frac"].astype(float).tolist(),
        context_rows=work,
    )
    rel_probs = probability_matrix(artifact.relevance_model, X, [0, 1])[:, 1]
    dir_probs = probability_matrix(artifact.direction_model, X, list(DIRECTION_LABELS))
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(work.itertuples(index=False)):
        direction_prob_map = {label: float(dir_probs[i, j]) for j, label in enumerate(DIRECTION_LABELS)}
        state = classify_action(float(rel_probs[i]), direction_prob_map, **artifact.thresholds)
        rows.append(
            {
                "target_action": str(row.action_label),
                "prediction": state["action"],
                "p_rate_relevant": float(rel_probs[i]),
                **{f"p_{k}": v for k, v in direction_prob_map.items()},
                "confidence": state["confidence"],
                "margin": state["margin"],
            }
        )
    return pd.DataFrame(rows)


def timeline_for_text(
    artifact: PolicySignalArtifact,
    text: str,
    *,
    speaker: str = "",
    context_rows: pd.DataFrame | None = None,
    min_words: int = 60,
    step_words: int = 100,
    max_points: int = 35,
) -> pd.DataFrame:
    words = str(text).split()
    if not words:
        return pd.DataFrame()
    total = len(words)
    points = list(range(min(int(min_words), total), total + 1, int(step_words)))
    if total not in points:
        points.append(total)
    if len(points) > int(max_points):
        idx = np.linspace(0, len(points) - 1, int(max_points)).round().astype(int)
        points = [points[i] for i in sorted(set(idx))]
        if points[-1] != total:
            points.append(total)

    base_context = context_rows
    if base_context is None:
        base_context = pd.DataFrame({"speaker_tier": [speaker_tier(speaker)]})

    rows: list[dict[str, Any]] = []
    for n_words in points:
        prefix = " ".join(words[:n_words])
        frac = n_words / max(1, total)
        X = transform_for_artifact(artifact, [prefix], [frac], context_rows=base_context)
        p_rel = float(probability_matrix(artifact.relevance_model, X, [0, 1])[0, 1])
        probs = probability_matrix(artifact.direction_model, X, list(DIRECTION_LABELS))[0]
        direction_prob_map = {label: float(probs[j]) for j, label in enumerate(DIRECTION_LABELS)}
        state = classify_action(p_rel, direction_prob_map, **artifact.thresholds)
        rows.append(
            {
                "speaker": speaker,
                "prefix_words": int(n_words),
                "prefix_pct": float(round(100 * frac, 1)),
                "prediction": state["action"],
                "state": state["state"],
                "p_rate_relevant": p_rel,
                "p_no_rate_signal": float(1 - p_rel),
                **{f"p_{k}": v for k, v in direction_prob_map.items()},
                "confidence": state["confidence"],
                "margin": state["margin"],
            }
        )
    return pd.DataFrame(rows)


def first_stable_correct_pct(timeline: pd.DataFrame, true_action: str, stable_steps: int = 2) -> float:
    if timeline.empty:
        return float("nan")
    preds = timeline["prediction"].astype(str).tolist()
    pcts = timeline["prefix_pct"].astype(float).tolist()
    for i in range(0, max(0, len(preds) - stable_steps + 1)):
        window = preds[i : i + stable_steps]
        if window and all(p == true_action for p in window):
            return float(pcts[i])
    return float("nan")


def evaluate_static_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    present_labels = [label for label in PRIMARY_ACTION_LABELS if label in set(y_true)]
    if not present_labels:
        present_labels = list(PRIMARY_ACTION_LABELS)
    return {
        "action_accuracy": float(accuracy_score(y_true, y_pred)),
        "action_f1_macro": float(f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)),
        "action_f1_macro_all_labels": float(
            f1_score(y_true, y_pred, labels=list(PRIMARY_ACTION_LABELS), average="macro", zero_division=0)
        ),
        "action_eval_labels": ",".join(present_labels),
        "uncertain_rate_signal_rate": float(np.mean([p == "uncertain_rate_signal" for p in y_pred])),
    }


def evaluate_online_speed(
    artifact: PolicySignalArtifact,
    val_docs: pd.DataFrame,
    *,
    step_pct: int = 10,
    stable_steps: int = 2,
    max_docs: int = 140,
) -> pd.DataFrame:
    sample = val_docs.sort_values("event_date").tail(int(max_docs)).copy()
    rows: list[dict[str, Any]] = []
    pct_grid = list(range(max(5, int(step_pct)), 101, max(5, int(step_pct))))
    for idx, row in sample.iterrows():
        timeline_rows: list[dict[str, Any]] = []
        for pct in pct_grid:
            prefix = prefix_text(str(row["document"]), pct)
            X = transform_for_artifact(artifact, [prefix], [pct / 100.0], context_rows=row.to_frame().T)
            p_rel = float(probability_matrix(artifact.relevance_model, X, [0, 1])[0, 1])
            probs = probability_matrix(artifact.direction_model, X, list(DIRECTION_LABELS))[0]
            direction_prob_map = {label: float(probs[j]) for j, label in enumerate(DIRECTION_LABELS)}
            state = classify_action(p_rel, direction_prob_map, **artifact.thresholds)
            timeline_rows.append({"prefix_pct": pct, "prediction": state["action"], "confidence": state["confidence"]})
        timeline = pd.DataFrame(timeline_rows)
        true_action = str(row["action_label"])
        stable_pct = first_stable_correct_pct(timeline, true_action, stable_steps=stable_steps)
        final_pred = str(timeline.iloc[-1]["prediction"]) if not timeline.empty else ""
        cue_pct = float(row.get("cue_pct", float("nan")))
        has_cue = not np.isnan(cue_pct)
        early_by_half = (not np.isnan(stable_pct)) and stable_pct <= 50.0
        before_cue = (not np.isnan(stable_pct)) and has_cue and stable_pct < cue_pct
        rows.append(
            {
                "row_id": int(idx),
                "event_date": row["event_date"],
                "source": row["source"],
                "speaker": row["speaker"],
                "true_action": true_action,
                "cue_pct": cue_pct,
                "first_stable_correct_pct": stable_pct,
                "early_by_50pct": bool(early_by_half),
                "correct_before_explicit_cue": bool(before_cue),
                "final_prediction": final_pred,
                "final_correct": bool(final_pred == true_action),
            }
        )
    return pd.DataFrame(rows)


def summarize_online_speed(speed_df: pd.DataFrame) -> dict[str, float]:
    if speed_df.empty:
        return {
            "online_final_accuracy": float("nan"),
            "online_early_by_50_rate": float("nan"),
            "online_before_cue_rate": float("nan"),
            "online_median_stable_correct_pct": float("nan"),
        }
    cue_mask = speed_df["cue_pct"].notna()
    stable = speed_df["first_stable_correct_pct"].dropna()
    return {
        "online_final_accuracy": float(speed_df["final_correct"].mean()),
        "online_early_by_50_rate": float(speed_df["early_by_50pct"].mean()),
        "online_before_cue_rate": float(speed_df.loc[cue_mask, "correct_before_explicit_cue"].mean()) if cue_mask.any() else float("nan"),
        "online_median_stable_correct_pct": float(stable.median()) if not stable.empty else float("nan"),
    }


def plot_confusion(y_true: list[str], y_pred: list[str], labels: list[str], out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_timeline(timeline: pd.DataFrame, out_path: Path, title: str = "Selected Speech Online Prediction") -> None:
    if timeline.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timeline["prefix_pct"], timeline["p_rate_relevant"], label="P(rate relevant)", color="#111827", linewidth=2)
    for label, color in [("lower", "#2563eb"), ("maintain", "#64748b"), ("raise", "#dc2626")]:
        ax.plot(timeline["prefix_pct"], timeline[f"p_{label}"], label=f"P({label})", linewidth=1.8, color=color)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Speech consumed (%)")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_leaderboard(leaderboard: pd.DataFrame, out_dir: Path) -> None:
    if leaderboard.empty:
        return
    top = leaderboard.sort_values("action_f1_macro", ascending=False).head(14).copy()
    top["model_label"] = top["family"] + "\n" + top["variant"].astype(str)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(top["model_label"], top["action_f1_macro"], label="Action macro-F1", color="#2563eb")
    ax.plot(top["model_label"], top["online_early_by_50_rate"], label="Early-by-half rate", color="#16a34a", marker="o")
    ax.plot(top["model_label"], top["online_final_accuracy"], label="Online final accuracy", color="#dc2626", marker="o")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=70)
    ax.set_title("Policy Signal Model Tuning Leaderboard")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "leaderboard_model_comparison.png", dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    ax2.scatter(leaderboard["online_median_stable_correct_pct"], leaderboard["action_f1_macro"], s=70, alpha=0.75)
    for _, row in leaderboard.iterrows():
        ax2.text(row["online_median_stable_correct_pct"], row["action_f1_macro"], str(row["family"])[:12], fontsize=7)
    ax2.set_xlim(0, 105)
    ax2.set_ylim(0, 1.02)
    ax2.invert_xaxis()
    ax2.set_xlabel("Median stable-correct point (% of speech, lower is faster)")
    ax2.set_ylabel("Action macro-F1")
    ax2.set_title("Accuracy vs Speed")
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(out_dir / "accuracy_vs_speed.png", dpi=160)
    plt.close(fig2)


def save_policy_artifact(artifact: PolicySignalArtifact, path: Path | None = None) -> Path:
    out = path or (ARTIFACTS_DIR / "policy_signal_tuning" / "latest" / "best_policy_signal_model.joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)
    return out


def load_policy_artifact(path: Path | None = None) -> PolicySignalArtifact | None:
    p = path or (ARTIFACTS_DIR / "policy_signal_tuning" / "latest" / "best_policy_signal_model.joblib")
    if not p.is_file():
        return None
    return joblib.load(p)
