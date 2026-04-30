"""Streamlit UI: corpus overview, time series, text analysis. Run: streamlit run streamlit_app.py"""

from __future__ import annotations

import sys
import re
import sqlite3
import base64
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import f1_score

from core.agentic_fed import (
    AgentUnavailableError,
    portkey_setup_status,
    run_fed_agent,
)
from core.config import AUDIT_DB, DATA_DIR
from core.analysis_pipeline import analyze_text
from core.ingest import (
    add_parsed_dates,
    build_combined_eda_frame,
    load_fomc,
    load_speaker,
)
from core.policy_signal_ml import (
    DIRECTION_LABELS,
    DOVISH_TERMS,
    HAWKISH_TERMS,
    POLICY_KEYWORDS,
    add_structured_policy_context,
    attach_claude_policy_features,
    attach_fred_context,
    first_stable_correct_pct,
    load_policy_artifact,
    make_policy_signal_frame,
    predict_action_frame,
    speaker_tier,
    timeline_for_text,
    weak_action_label,
)
from components.fed_tracker import render_fed_tracker

st.set_page_config(page_title="Fed Speech Early-Warning Dashboard", layout="wide")

# Sticky header CSS for Fed tracker
st.markdown("""
<style>
div[data-testid="stVerticalBlock"] div:has(div.fed-tracker-sticky) {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: var(--background-color);
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)
# Sticky Fed Tracker (visible on all tabs)
render_fed_tracker()
st.markdown('<div class="fed-tracker-sticky"/>', unsafe_allow_html=True)
st.markdown("---")

st.title("Fed Speech Early-Warning Dashboard")
st.caption(
    "Select a Fed speech, let the model read it in chunks, and see how early it can infer "
    "whether the text points to a rate cut, a hold, a hike, or no clear rate signal."
)

tab_policy, tab_data, tab_analyze, tab_api = st.tabs(
    ["Speech Early-Warning ML", "Corpus & time series", "Analyze custom text", "Call Flask API"]
)


def _step_words_for_text(text: str) -> int:
    return max(80, int(max(1, len(str(text).split())) / 24))


LABEL_COPY = {
    "lower": {
        "short": "Cut",
        "badge": "LIKELY CUT",
        "sentence": "The model thinks this text leans toward lower interest rates.",
        "color": "#16a34a",
        "bg": "#dcfce7",
    },
    "maintain": {
        "short": "Hold",
        "badge": "LIKELY HOLD",
        "sentence": "The model thinks the Fed is most likely to hold rates steady.",
        "color": "#4b5563",
        "bg": "#e5e7eb",
    },
    "raise": {
        "short": "Hike",
        "badge": "LIKELY HIKE",
        "sentence": "The model thinks this text leans toward higher interest rates.",
        "color": "#dc2626",
        "bg": "#fee2e2",
    },
    "no_rate_signal": {
        "short": "No clear rate signal",
        "badge": "NO CLEAR RATE SIGNAL",
        "sentence": "The model thinks this text is probably not about the next rate decision.",
        "color": "#334155",
        "bg": "#e2e8f0",
    },
    "uncertain_rate_signal": {
        "short": "Not confident",
        "badge": "NOT CONFIDENT",
        "sentence": "The model was not confident enough to make a call. This speech may not be about rate decisions, or the signal may be mixed.",
        "color": "#92400e",
        "bg": "#fef3c7",
    },
    "unavailable": {
        "short": "Unavailable",
        "badge": "NO MODEL RESULT",
        "sentence": "The model could not produce a prediction for this text.",
        "color": "#334155",
        "bg": "#f1f5f9",
    },
}


PROBABILITY_LABELS = {
    "no_rate_signal": "No clear rate signal",
    "lower": "Cut rates",
    "maintain": "Hold steady",
    "raise": "Hike rates",
    "uncertain_rate_signal": "Not confident",
}


def _label_info(label: object) -> dict[str, str]:
    return LABEL_COPY.get(str(label or "unavailable"), LABEL_COPY["unavailable"])


def _plain_label(label: object) -> str:
    return _label_info(label)["short"]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _confidence_band(confidence: object) -> str:
    score = _safe_float(confidence)
    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Moderate"
    return "Low"


def _render_prediction_badge(label: object, *, prefix: str = "Verdict") -> None:
    info = _label_info(label)
    st.markdown(
        f"""
        <div style="margin:.4rem 0 1rem 0;">
          <div style="font-size:.8rem; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.05em;">{prefix}</div>
          <div style="display:inline-block; padding:.85rem 1.1rem; border-radius:999px; background:{info['bg']};
                      color:{info['color']}; font-size:1.55rem; font-weight:900; border:1px solid rgba(15,23,42,.08);">
            {info['badge']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_confidence_gauge(confidence: object) -> None:
    score = max(0.0, min(1.0, _safe_float(confidence)))
    pct = score * 100
    band = _confidence_band(score)
    st.markdown(
        f"""
        <div style="margin:.4rem 0 1rem 0;">
          <div style="display:flex; justify-content:space-between; font-weight:750; margin-bottom:.25rem;">
            <span>Confidence: {band}</span><span>{pct:.0f}%</span>
          </div>
          <div style="position:relative; height:22px; border-radius:999px;
                      background:linear-gradient(90deg,#fee2e2 0%,#fee2e2 55%,#fef3c7 55%,#fef3c7 75%,#dcfce7 75%,#dcfce7 100%);
                      border:1px solid #d1d5db;">
            <div style="position:absolute; left:calc({pct:.2f}% - 6px); top:-4px; width:12px; height:30px;
                        border-radius:999px; background:#111827; box-shadow:0 1px 4px rgba(0,0,0,.25);"></div>
          </div>
          <div style="display:flex; justify-content:space-between; color:#64748b; font-size:.8rem; margin-top:.15rem;">
            <span>Low</span><span>Moderate</span><span>High</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_featured_visuals_strip() -> None:
    st.markdown("### Start here: the visuals that explain the product")
    st.write(
        "These are the presentation-friendly views: first confirm the speech is rate-relevant, "
        "then watch the model's cut / hold / hike call evolve as more words arrive."
    )
    st.markdown(
        """
        <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.75rem; margin:.35rem 0 1rem 0;">
          <div style="border:2px solid #bbf7d0; border-radius:18px; padding:.95rem; background:#f0fdf4; box-shadow:0 8px 22px rgba(21,128,61,.10);">
            <div style="font-size:.76rem; font-weight:900; color:#15803d; letter-spacing:.06em; text-transform:uppercase;">First gate</div>
            <div style="font-weight:900; color:#15803d; font-size:1.05rem; margin-top:.15rem;">Rate relevance scanner</div>
            <div style="font-size:.9rem; color:#334155; margin-top:.25rem;">Shows whether this is actually about policy rates before trusting a cut / hold / hike call.</div>
          </div>
          <div style="border:2px solid #dbeafe; border-radius:18px; padding:.95rem; background:#eff6ff; box-shadow:0 8px 22px rgba(37,99,235,.09);">
            <div style="font-size:.76rem; font-weight:900; color:#1d4ed8; letter-spacing:.06em; text-transform:uppercase;">Main story</div>
            <div style="font-weight:900; color:#1d4ed8; font-size:1.05rem; margin-top:.15rem;">Prediction timeline</div>
            <div style="font-size:.9rem; color:#334155; margin-top:.25rem;">Shows how early the model starts leaning cut, hold, or hike as the speech unfolds.</div>
          </div>
          <div style="border:2px solid #fed7aa; border-radius:18px; padding:.95rem; background:#fff7ed; box-shadow:0 8px 22px rgba(194,65,12,.08);">
            <div style="font-size:.76rem; font-weight:900; color:#c2410c; letter-spacing:.06em; text-transform:uppercase;">Why it thinks that</div>
            <div style="font-weight:900; color:#c2410c; font-size:1.05rem; margin-top:.15rem;">Hawkish vs. dovish words</div>
            <div style="font-size:.9rem; color:#334155; margin-top:.25rem;">Turns the model's reasoning into visible language patterns anyone can understand.</div>
          </div>
          <div style="border:2px solid #e9d5ff; border-radius:18px; padding:.95rem; background:#faf5ff; box-shadow:0 8px 22px rgba(126,34,206,.08);">
            <div style="font-size:.76rem; font-weight:900; color:#7e22ce; letter-spacing:.06em; text-transform:uppercase;">Big-picture read</div>
            <div style="font-weight:900; color:#7e22ce; font-size:1.05rem; margin-top:.15rem;">Macro forecast</div>
            <div style="font-size:.9rem; color:#334155; margin-top:.25rem;">Aggregates recent Fed communications into one cut / hold / hike lean.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _term_counts(text: str, terms: tuple[str, ...]) -> dict[str, int]:
    lower = str(text).lower()
    counts = {}
    for term in terms:
        n = len(re.findall(re.escape(term.lower()), lower))
        if n:
            counts[term] = n
    return counts


def _tone_counts(text: str) -> tuple[dict[str, int], dict[str, int]]:
    return _term_counts(text, HAWKISH_TERMS), _term_counts(text, DOVISH_TERMS)


def _policy_keyword_counts(text: str) -> dict[str, int]:
    lower = str(text).lower()
    counts = {}
    for term in POLICY_KEYWORDS:
        n = len(re.findall(re.escape(str(term).lower()), lower))
        if n:
            counts[str(term)] = n
    return counts


def _rate_relevance_label(score: float) -> tuple[str, str, str]:
    if score >= 0.75:
        return "Strong rate-policy signal", "#16a34a", "#dcfce7"
    if score >= 0.50:
        return "Moderate rate-policy signal", "#ca8a04", "#fef9c3"
    if score >= 0.30:
        return "Weak / mixed rate-policy signal", "#f97316", "#ffedd5"
    return "Probably not a rate-decision speech", "#64748b", "#e2e8f0"


def _render_rate_relevance_scanner(
    text: str,
    rate_relevance: object | None,
    no_signal_probability: object | None = None,
) -> None:
    score = max(0.0, min(1.0, _safe_float(rate_relevance)))
    pct = 100 * score
    no_signal = _safe_float(no_signal_probability, default=1.0 - score)
    no_signal = max(0.0, min(1.0, no_signal))
    gate_delta = score - no_signal
    label, color, bg = _rate_relevance_label(score)
    policy_counts = _policy_keyword_counts(text)
    total_policy_hits = sum(policy_counts.values())
    top_terms = pd.Series(policy_counts, dtype=float).sort_values(ascending=False).head(12)

    st.markdown("### Is this actually about interest rates?")
    st.markdown(
        f"""
        <div style="border:1px solid #dbeafe; border-radius:18px; padding:1rem; background:#f8fafc; margin-bottom:.75rem;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;">
            <div>
              <div style="font-size:.8rem; color:#64748b; font-weight:800; letter-spacing:.06em; text-transform:uppercase;">Rate relevance scanner</div>
              <div style="font-size:1.35rem; font-weight:900; color:{color}; margin-top:.2rem;">{label}</div>
            </div>
            <div style="padding:.55rem .8rem; border-radius:999px; background:{bg}; color:{color}; font-weight:900;">{pct:.0f}% rate-relevant</div>
          </div>
          <div style="margin-top:.8rem; height:24px; border-radius:999px; background:linear-gradient(90deg,#e2e8f0 0%,#e2e8f0 30%,#ffedd5 30%,#ffedd5 50%,#fef9c3 50%,#fef9c3 75%,#dcfce7 75%,#dcfce7 100%); border:1px solid #cbd5e1; position:relative;">
            <div style="position:absolute; left:calc({pct:.2f}% - 7px); top:-5px; width:14px; height:34px; border-radius:999px; background:#0f172a; box-shadow:0 1px 5px rgba(15,23,42,.35);"></div>
          </div>
          <div style="display:flex; justify-content:space-between; color:#64748b; font-size:.78rem; margin-top:.2rem;">
            <span>Not rate-related</span><span>Mixed</span><span>Probably rate-related</span><span>Strong signal</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Model's first decision gate: relevant or not?**")
    gate_df = pd.Series(
        {
            "Speech looks rate-related": score,
            "Speech looks off-topic / no clear rate signal": no_signal,
        }
    )
    st.bar_chart(gate_df)
    gate_message = (
        "The model should continue to the cut / hold / hike question."
        if gate_delta >= 0
        else "The model thinks this may be a weak rate-policy speech, so the direction call needs caution."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rate-policy phrase hits", int(total_policy_hits))
    c2.metric("Speech length", f"{len(str(text).split()):,} words")
    c3.metric("Decision gate", "Analyze direction" if score >= 0.50 else "Treat cautiously")
    c4.metric("Relevance edge", f"{100 * gate_delta:+.0f} pts")
    st.caption(
        f"{gate_message} The scanner combines the trained relevance probability with visible policy-rate phrases below."
    )
    if not top_terms.empty:
        fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.34 * len(top_terms))))
        ax.barh(top_terms.index, top_terms.values, color="#2563eb")
        ax.invert_yaxis()
        ax.set_xlabel("Times phrase appears")
        ax.set_title("Policy phrases supporting rate relevance")
        for i, value in enumerate(top_terms.values):
            ax.text(float(value) + 0.05, i, str(int(value)), va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No tracked policy-rate phrases were found, so any rate-direction prediction should be treated cautiously.")


def _what_it_means(label: object, text: str, confidence: object, rate_relevance: object | None = None) -> str:
    hawk, dove = _tone_counts(text)
    hawk_total = sum(hawk.values())
    dove_total = sum(dove.values())
    relevance = _safe_float(rate_relevance, default=float("nan"))
    relevance_phrase = ""
    if not pd.isna(relevance):
        if relevance >= 0.70:
            relevance_phrase = " The model also sees this as strongly related to interest-rate policy."
        elif relevance >= 0.40:
            relevance_phrase = " The model sees some rate-policy relevance, but the signal is not overwhelming."
        else:
            relevance_phrase = " The model sees weak rate-policy relevance, so the result should be treated cautiously."
    return (
        f"{_label_info(label)['sentence']} It found {hawk_total} hawkish phrase hits and "
        f"{dove_total} dovish phrase hits, then combined those language clues with macro context and speaker information."
        f"{relevance_phrase} Confidence is {_confidence_band(confidence).lower()}, so this is a model signal rather than a guarantee."
    )


def _render_explanation(label: object, text: str, confidence: object, rate_relevance: object | None = None) -> None:
    st.markdown("### What does this mean?")
    st.info(_what_it_means(label, text, confidence, rate_relevance))


def _render_tone_breakdown(text: str) -> None:
    hawk, dove = _tone_counts(text)
    rows = (
        [{"term": term, "count": count, "tone": "Hawkish / higher-rate language"} for term, count in hawk.items()]
        + [{"term": term, "count": count, "tone": "Dovish / lower-rate language"} for term, count in dove.items()]
    )
    if not rows:
        st.caption("No tracked hawkish or dovish keyword phrases were found in this text.")
        return
    df = pd.DataFrame(rows).sort_values("count", ascending=False).head(20)
    colors = df["tone"].map(
        {
            "Hawkish / higher-rate language": "#dc2626",
            "Dovish / lower-rate language": "#16a34a",
        }
    )
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.36 * len(df))))
    ax.barh(df["term"], df["count"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Times phrase appears")
    ax.set_title("Hawkish vs. dovish language the model can see")
    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(float(row.count) + 0.05, i, str(int(row.count)), va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _first_stable_call(timeline: pd.DataFrame, stable_steps: int = 2) -> tuple[float | None, str | None]:
    if timeline.empty:
        return None, None
    preds = timeline["prediction"].astype(str).tolist()
    pcts = timeline["prefix_pct"].astype(float).tolist()
    for i in range(0, max(0, len(preds) - stable_steps + 1)):
        window = preds[i : i + stable_steps]
        if window and len(set(window)) == 1 and window[0] not in {"uncertain_rate_signal"}:
            return float(pcts[i]), window[0]
    return None, None


def _render_timeline_chart(timeline: pd.DataFrame, *, target: str | None = None, stable_pct: float | None = None) -> None:
    if timeline.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.axhspan(0.5, 1.0, color="#dcfce7", alpha=0.35, label="Above 50% = stronger signal")
    ax.axhspan(0.0, 0.5, color="#fee2e2", alpha=0.18)
    ax.axhline(0.5, color="#111827", linestyle="--", linewidth=1.2)
    series = [
        ("p_rate_relevant", "Rate-relevant", "#111827", 2.3),
        ("p_lower", "Cut", "#16a34a", 2.0),
        ("p_maintain", "Hold", "#4b5563", 2.0),
        ("p_raise", "Hike", "#dc2626", 2.0),
    ]
    for col, label, color, width in series:
        if col in timeline.columns:
            ax.plot(timeline["prefix_pct"], timeline[col], label=label, color=color, linewidth=width)

    call_pct, call_label = _first_stable_call(timeline)
    annotation_pct = stable_pct if stable_pct is not None and pd.notna(stable_pct) else call_pct
    annotation_label = target if stable_pct is not None and pd.notna(stable_pct) else call_label
    if annotation_pct is not None:
        ax.axvline(float(annotation_pct), color="#f59e0b", linestyle=":", linewidth=2.3)
        ax.annotate(
            f"First stable call: {_plain_label(annotation_label)} at {float(annotation_pct):.0f}%",
            xy=(float(annotation_pct), 0.88),
            xytext=(min(82, float(annotation_pct) + 4), 0.92),
            arrowprops={"arrowstyle": "->", "color": "#92400e"},
            fontsize=9,
            color="#92400e",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#fef3c7", "ec": "#f59e0b", "alpha": 0.95},
        )
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent of speech read")
    ax.set_ylabel("Model probability")
    ax.set_title("How the model's rate signal changes as more of the speech is read")
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _display_probability_series(probs: pd.Series | dict[str, float]) -> pd.Series:
    s = pd.Series(probs, dtype=float)
    return s.rename(index=PROBABILITY_LABELS)


def _render_shap_highlighted_text(text: str, top_features: list[dict[str, Any]], max_words: int = 800) -> None:
    """Render text with SHAP-style highlighting based on model feature contributions.
    
    Highlights important words:
    - Green background: positive contribution (pushes toward predicted class)
    - Red background: negative contribution (pushes away from predicted class)
    - Intensity scales with absolute contribution magnitude
    """
    if not top_features or not text.strip():
        st.markdown(f'<div style="background:#f8f9fa;padding:1rem;border-radius:8px;max-height:400px;overflow-y:auto;line-height:1.6;white-space:pre-wrap;font-family:system-ui,-apple-system,sans-serif;font-size:0.95rem;">{text[:5000]}</div>', unsafe_allow_html=True)
        return
    
    # Build token → contribution map (case-insensitive)
    token_contrib = {}
    max_abs = 0.0
    for feat in top_features:
        token = str(feat.get("token", "")).lower()
        contrib = float(feat.get("contribution", 0))
        if token and contrib != 0:
            token_contrib[token] = contrib
            max_abs = max(max_abs, abs(contrib))
    
    if not token_contrib or max_abs == 0:
        st.markdown(f'<div style="background:#f8f9fa;padding:1rem;border-radius:8px;max-height:400px;overflow-y:auto;line-height:1.6;white-space:pre-wrap;font-family:system-ui,-apple-system,sans-serif;font-size:0.95rem;">{text[:5000]}</div>', unsafe_allow_html=True)
        return
    
    # Tokenize text (simple whitespace + punctuation split)
    import re
    words = re.findall(r'\S+', text[:max_words * 8])  # rough char estimate
    if len(words) > max_words:
        words = words[:max_words]
        truncated = True
    else:
        truncated = False
    
    # Build highlighted HTML
    html_parts = []
    for word in words:
        word_clean = re.sub(r'[^\w]', '', word.lower())
        if word_clean in token_contrib:
            contrib = token_contrib[word_clean]
            intensity = abs(contrib) / max_abs
            # Scale opacity 0.2 to 0.8 based on intensity
            opacity = 0.2 + (intensity * 0.6)
            if contrib > 0:
                # Positive contribution → green
                color = f"rgba(34, 197, 94, {opacity})"
            else:
                # Negative contribution → red
                color = f"rgba(239, 68, 68, {opacity})"
            html_parts.append(f'<span style="background:{color};padding:2px 4px;border-radius:3px;margin:0 2px;">{word}</span>')
        else:
            html_parts.append(word)
    
    highlighted = " ".join(html_parts)
    if truncated:
        highlighted += ' <span style="color:#9ca3af;font-style:italic;">... (text truncated for display)</span>'
    
    st.markdown(
        f'<div style="background:#f8f9fa;padding:1.2rem;border-radius:8px;max-height:450px;overflow-y:auto;line-height:1.7;font-family:system-ui,-apple-system,sans-serif;font-size:0.95rem;">'
        f'<div style="margin-bottom:0.5rem;font-size:0.85rem;color:#6b7280;font-weight:500;">📊 SHAP-style feature importance: <span style="background:rgba(34,197,94,0.5);padding:2px 6px;border-radius:3px;margin:0 4px;">green = positive</span> <span style="background:rgba(239,68,68,0.5);padding:2px 6px;border-radius:3px;">red = negative</span></div>'
        f'{highlighted}'
        f'</div>',
        unsafe_allow_html=True
    )


def _calculate_text_area_height(text: str, min_height: int = 120, max_height: int = 600) -> int:
    """Calculate appropriate text area height based on content."""
    if not text:
        return min_height
    line_count = text.count('\n') + 1
    # Estimate wrapped lines (assume ~80 chars per line)
    char_count = len(text)
    estimated_wrapped_lines = max(line_count, char_count // 80)
    # ~20px per line + padding
    calculated = estimated_wrapped_lines * 22 + 40
    return max(min_height, min(calculated, max_height))


def _render_policy_timeline(
    artifact,
    text: str,
    *,
    speaker: str = "custom",
    target: str | None = None,
    context_rows: pd.DataFrame | None = None,
) -> None:
    timeline = timeline_for_text(
        artifact,
        text,
        speaker=speaker,
        context_rows=context_rows,
        min_words=60,
        step_words=_step_words_for_text(text),
        max_points=30,
    )
    if timeline.empty:
        st.warning("Not enough usable text to generate a policy-signal timeline.")
        return

    final = timeline.iloc[-1]
    stable_pct = first_stable_correct_pct(timeline, target, stable_steps=2) if target else None
    pred_label = str(final["prediction"])
    st.markdown(f"## {_label_info(pred_label)['sentence']}")
    _render_prediction_badge(pred_label)
    _render_confidence_gauge(final["confidence"])
    r1, r2, r3 = st.columns(3)
    r1.metric("Words analyzed", f"{len(str(text).split()):,}")
    r2.metric("Rate relevance", _confidence_band(final["p_rate_relevant"]))
    r3.metric("Final speech progress", "100%")
    _render_rate_relevance_scanner(text, final.get("p_rate_relevant"), final.get("p_no_rate_signal"))
    _render_explanation(pred_label, text, final["confidence"], final["p_rate_relevant"])

    # Get top features for SHAP highlighting
    from core.analysis_pipeline import analyze_text
    analysis_result = analyze_text(text)
    top_features = analysis_result.get("explanation_top_features") or []
    if top_features:
        st.markdown("### 📝 Speech with model feature importance")
        st.caption("Words highlighted by their contribution to the prediction (brighter = stronger influence)")
        _render_shap_highlighted_text(text, top_features, max_words=700)

    if target:
        if pd.notna(stable_pct):
            st.success(
                f"For this historical speech, the mapped outcome was **{_plain_label(target)}**. "
                f"The model first became stably correct after reading about **{float(stable_pct):.1f}%** of the speech."
            )
        else:
            st.warning(
                f"For this historical speech, the mapped outcome was **{_plain_label(target)}**. "
                "The model did not become stably correct for that label."
            )

    st.markdown("### How the prediction changed as the speech unfolded")
    _render_timeline_chart(timeline, target=target, stable_pct=stable_pct)
    st.markdown("### Hawkish vs. dovish language found")
    _render_tone_breakdown(text)
    with st.expander("Detailed prefix table", expanded=False):
        readable = timeline.copy()
        readable["prediction_plain_english"] = readable["prediction"].map(_plain_label)
        readable = readable.drop(columns=[col for col in ["prediction", "state"] if col in readable.columns])
        readable = readable.rename(
            columns={
                "prefix_pct": "percent_of_speech_read",
                "p_rate_relevant": "probability_rate_related",
                "p_no_rate_signal": "probability_no_clear_rate_signal",
                "p_lower": "probability_cut",
                "p_maintain": "probability_hold",
                "p_raise": "probability_hike",
            }
        )
        st.dataframe(readable, use_container_width=True)

    hawk, dove = _tone_counts(text)
    call_pct, call_label = _first_stable_call(timeline)
    agent_context = {
        "result_type": "speech_early_warning",
        "final_prediction": pred_label,
        "prediction_plain_english": _plain_label(pred_label),
        "confidence": float(final.get("confidence", 0.0)),
        "rate_relevance": float(final.get("p_rate_relevant", 0.0)),
        "probabilities": {
            "no_rate_signal": float(final.get("p_no_rate_signal", 0.0)),
            "lower": float(final.get("p_lower", 0.0)),
            "maintain": float(final.get("p_maintain", 0.0)),
            "raise": float(final.get("p_raise", 0.0)),
        },
        "target_label_if_historical": target,
        "first_stable_call": {"prediction": call_label, "prefix_pct": call_pct},
        "stable_correct_pct_if_historical": stable_pct,
        "hawkish_phrase_hits": sum(hawk.values()),
        "dovish_phrase_hits": sum(dove.values()),
        "word_count": len(str(text).split()),
        "timeline_tail": timeline.tail(8).to_dict(orient="records"),
    }
    _render_result_agent_chat("policy_result", agent_context, text)


def _macro_forecast_from_texts(
    artifact,
    texts: list[str],
    context_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    contexts = context_rows.reset_index(drop=True) if context_rows is not None else None
    for idx, text in enumerate(texts):
        if len(str(text).split()) < 80:
            continue
        row_context = contexts.iloc[[idx]] if contexts is not None and idx < len(contexts) else None
        timeline = timeline_for_text(
            artifact,
            str(text),
            speaker="macro",
            context_rows=row_context,
            min_words=60,
            step_words=_step_words_for_text(str(text)),
            max_points=24,
        )
        if timeline.empty:
            continue
        final = timeline.iloc[-1]
        rows.append(
            {
                "lower": float(final["p_lower"]),
                "maintain": float(final["p_maintain"]),
                "raise": float(final["p_raise"]),
                "rate_relevant": float(final["p_rate_relevant"]),
                "prediction": str(final["prediction"]),
            }
        )
    return pd.DataFrame(rows)


def _context_for_document_rows(rows: pd.DataFrame) -> pd.DataFrame:
    participant = rows["participant"] if "participant" in rows.columns else pd.Series([""] * len(rows), index=rows.index)
    domain = rows["domain"] if "domain" in rows.columns else pd.Series([""] * len(rows), index=rows.index)
    source = rows["source"] if "source" in rows.columns else pd.Series(["speaker"] * len(rows), index=rows.index)
    context = pd.DataFrame(
        {
            "event_date": pd.to_datetime(
                rows["date"].astype(str).str.replace(r"\.0$", "", regex=True),
                format="%Y%m%d",
                errors="coerce",
            ),
            "speaker_tier": [
                speaker_tier(name, source=str(src), domain=str(dom))
                for name, src, dom in zip(participant, source, domain)
            ],
            "source": source.astype(str).to_numpy(),
            "speaker": participant.astype(str).to_numpy(),
            "domain": domain.astype(str).to_numpy(),
            "document": rows["document"].astype(str).to_numpy() if "document" in rows.columns else [""] * len(rows),
            "high": rows["high"].to_numpy() if "high" in rows.columns else [0] * len(rows),
            "low": rows["low"].to_numpy() if "low" in rows.columns else [0] * len(rows),
            "fomc_ref_date": (
                rows["fomc-ref-date"].astype(str).to_numpy()
                if "fomc-ref-date" in rows.columns
                else rows["date"].astype(str).to_numpy()
            ),
            "alignment_rule": rows["alignment_rule"].astype(str).to_numpy() if "alignment_rule" in rows.columns else [""] * len(rows),
            "quality_score": rows["quality_score"].to_numpy() if "quality_score" in rows.columns else [1.0] * len(rows),
            "labels_from_fred": rows["labels_from_fred"].to_numpy() if "labels_from_fred" in rows.columns else [False] * len(rows),
        }
    )
    context = add_structured_policy_context(context)
    context = attach_claude_policy_features(context)
    return attach_fred_context(context)


def _context_for_speech_rows(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    work["source"] = "speaker"
    return _context_for_document_rows(work)


def _load_fred_history(series_id: str = "FEDFUNDS") -> pd.DataFrame:
    if not AUDIT_DB.is_file():
        return pd.DataFrame()
    try:
        with sqlite3.connect(AUDIT_DB) as conn:
            df = pd.read_sql_query(
                "SELECT obs_date, value FROM fred_cache WHERE series_id = ? ORDER BY obs_date",
                conn,
                params=(series_id,),
            )
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["obs_date", "value"])


def _render_fred_prediction_overlay(artifact) -> None:
    fred = _load_fred_history("FEDFUNDS")
    if fred.empty or artifact is None:
        st.info("FRED rate history or the tuned model artifact is unavailable.")
        return
    try:
        frame = make_policy_signal_frame(min_words=80)
        fomc_docs = frame[frame["source"].eq("fomc")].sort_values("event_date").tail(90).reset_index(drop=True)
        if fomc_docs.empty:
            st.info("No FOMC documents were available for the overlay.")
            return
        preds = predict_action_frame(artifact, fomc_docs, prefix_pct=100).reset_index(drop=True)
    except Exception as exc:
        st.warning("Could not build the FRED prediction overlay from local corpus data.")
        st.caption(str(exc))
        return

    overlay = fomc_docs[["event_date", "action_label"]].copy().reset_index(drop=True)
    overlay["prediction"] = preds["prediction"].astype(str)
    overlay["correct"] = overlay["prediction"].eq(overlay["action_label"])
    rate_line = fred.sort_values("obs_date").copy()
    overlay = pd.merge_asof(
        overlay.sort_values("event_date"),
        rate_line.rename(columns={"obs_date": "event_date", "value": "fedfunds"}).sort_values("event_date"),
        on="event_date",
        direction="backward",
    ).dropna(subset=["fedfunds"])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    recent_start = overlay["event_date"].min() - pd.Timedelta(days=90)
    rate_recent = rate_line[rate_line["obs_date"] >= recent_start]
    ax.plot(rate_recent["obs_date"], rate_recent["value"], color="#111827", linewidth=2.2, label="Effective federal funds rate")
    for is_correct, color, label in [(True, "#16a34a", "Model matched outcome"), (False, "#dc2626", "Model missed outcome")]:
        pts = overlay[overlay["correct"].eq(is_correct)]
        if not pts.empty:
            ax.scatter(pts["event_date"], pts["fedfunds"], s=64, color=color, edgecolor="white", linewidth=0.8, label=label, zorder=3)
    ax.set_title("Rate history with model predictions overlaid")
    ax.set_ylabel("Effective federal funds rate (%)")
    ax.set_xlabel("Document / meeting date")
    ax.grid(alpha=0.18)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Dots are historical FOMC communication examples scored by the current model. "
        "Green means the model's cut/hold/hike call matched the mapped outcome; red means it missed. "
        "Older FOMC examples may be partly in-sample, so use this as an explanatory visual rather than a trading backtest."
    )


def _speaker_role(value: object) -> str:
    text = str(value or "").lower()
    if "chair powell" in text or "jerome" in text and "powell" in text:
        return "Chair"
    if "vice chair" in text:
        return "Vice Chair"
    if "governor" in text:
        return "Governor"
    if "president" in text:
        return "Regional President"
    return "Other / unclear"


def _render_plain_confusion_matrix(latest_dir: Path) -> None:
    path = latest_dir / "best_validation_predictions.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    if not {"target_action", "prediction"}.issubset(df.columns):
        return
    row_order = ["no_rate_signal", "lower", "maintain", "raise"]
    col_order = ["no_rate_signal", "lower", "maintain", "raise", "uncertain_rate_signal"]
    cm = pd.crosstab(df["target_action"], df["prediction"]).reindex(index=row_order, columns=col_order, fill_value=0)
    row_labels = [_plain_label(x) for x in cm.index]
    col_labels = [_plain_label(x) for x in cm.columns]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(col_labels)), [f"Predicted: {x}" for x in col_labels], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), [f"Actual: {x}" for x in row_labels])
    ax.set_title("Model performance: correct predictions are on the diagonal")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm.values[i, j])), ha="center", va="center", color="#111827", fontweight="700")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Boxes on the diagonal mean the model predicted the same outcome that was mapped historically.")


def _render_temporal_f1(speed: pd.DataFrame) -> None:
    if not {"event_date", "true_action", "final_prediction"}.issubset(speed.columns):
        return
    work = speed.copy()
    work["event_year"] = pd.to_datetime(work["event_date"], errors="coerce").dt.year
    rows = []
    for year, group in work.dropna(subset=["event_year"]).groupby("event_year"):
        present = [label for label in ["no_rate_signal", "lower", "maintain", "raise"] if label in set(group["true_action"].astype(str))]
        if not present:
            continue
        rows.append(
            {
                "year": int(year),
                "action_macro_f1": float(
                    f1_score(
                        group["true_action"].astype(str),
                        group["final_prediction"].astype(str),
                        labels=present,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "n_docs": int(len(group)),
            }
        )
    if not rows:
        return
    drift = pd.DataFrame(rows).sort_values("year")
    st.markdown("**Temporal drift: action F1 by year**")
    st.line_chart(drift.set_index("year")["action_macro_f1"])
    st.caption("This shows whether the online model generalizes equally well across time. Drops usually mean the Fed's language regime changed.")
    with st.expander("Temporal drift table", expanded=False):
        st.dataframe(drift, use_container_width=True)


def _render_speaker_role_accuracy(speed: pd.DataFrame) -> None:
    if not {"speaker", "final_correct"}.issubset(speed.columns):
        return
    work = speed.copy()
    work["speaker_role"] = work["speaker"].map(_speaker_role)
    role_perf = (
        work.groupby("speaker_role")["final_correct"]
        .agg(accuracy="mean", n_docs="count")
        .sort_values(["accuracy", "n_docs"], ascending=False)
    )
    if role_perf.empty:
        return
    st.markdown("**Accuracy by speaker role**")
    st.bar_chart(role_perf["accuracy"])
    st.caption("This asks a very practical question: are Chair/Vice Chair communications easier for the model to read than regional speeches?")
    with st.expander("Speaker-role accuracy table", expanded=False):
        st.dataframe(role_perf, use_container_width=True)


def _context_id(ml_context: dict[str, object], source_text: str) -> str:
    return "|".join(
        [
            str(ml_context.get("result_type", "")),
            str(ml_context.get("final_prediction", ml_context.get("prediction_decision", ""))),
            str(round(_safe_float(ml_context.get("confidence", ml_context.get("prediction_confidence", 0))), 4)),
            str(len(str(source_text).split())),
        ]
    )


def _local_result_agent_answer(question: str, ml_context: dict[str, object]) -> str:
    prediction = ml_context.get("final_prediction") or ml_context.get("prediction_decision") or "unavailable"
    confidence = _safe_float(ml_context.get("confidence", ml_context.get("prediction_confidence", 0)))
    relevance = _safe_float(ml_context.get("rate_relevance", ml_context.get("prediction_rate_relevance", 0)))
    probabilities = ml_context.get("probabilities") or ml_context.get("prediction_proba") or {}
    stable = ml_context.get("first_stable_call") or ml_context.get("first_stable_prediction")
    hawk_hits = int(_safe_float(ml_context.get("hawkish_phrase_hits", 0)))
    dove_hits = int(_safe_float(ml_context.get("dovish_phrase_hits", 0)))
    result_type = str(ml_context.get("result_type", "model_result")).replace("_", " ")

    lines = [
        f"Here is the analyst read on this {result_type}:",
        "",
        f"- The model's main call is **{_plain_label(prediction)}**.",
        f"- Confidence is **{_confidence_band(confidence).lower()}** ({confidence:.3f}).",
        f"- Rate relevance is **{_confidence_band(relevance).lower()}** ({relevance:.3f}), which is the first gate before trusting the cut / hold / hike call.",
    ]
    if isinstance(probabilities, dict) and probabilities:
        pretty_probs = {
            PROBABILITY_LABELS.get(str(k), str(k)): round(_safe_float(v), 3)
            for k, v in probabilities.items()
        }
        lines.append(f"- Probability breakdown: **{pretty_probs}**.")
    if stable:
        lines.append(f"- Earliest stable call information: **{stable}**.")
    if hawk_hits or dove_hits:
        lines.append(f"- The visible language scan found **{hawk_hits} hawkish** phrase hits and **{dove_hits} dovish** phrase hits.")
    lines.extend(
        [
            "",
            "Plain-English interpretation:",
            _what_it_means(prediction, "", confidence, relevance),
            "",
            "Limitations to mention if presenting this:",
            "- This is a model signal, not a guaranteed Fed decision or trading recommendation.",
            "- Rate relevance matters: a low-relevance speech can produce a direction-looking probability that should not be trusted much.",
            "- The model can struggle during new policy regimes because Fed language changes across cycles.",
        ]
    )
    if question.strip():
        lines.extend(
            [
                "",
                f"Your question was: **{question.strip()}**",
                "Best answer: interpret the result through the two-step gate: first relevance, then direction. If relevance is weak, explain that the model is intentionally cautious.",
            ]
        )
    return "\n".join(lines)


def _run_inline_agent(
    question: str,
    *,
    ml_context: dict[str, object],
    source_text: str,
    previous_messages: list[dict[str, str]],
) -> tuple[str, list[dict[str, object]], bool]:
    status = portkey_setup_status()
    use_portkey = bool(status["portkey_ai_installed"] and status["api_key_configured"])
    if not use_portkey:
        return _local_result_agent_answer(question, ml_context), [], False

    history = "\n".join(f"{m['role']}: {m['content']}" for m in previous_messages[-6:])
    full_question = (
        f"{question}\n\nPrior chat about this model result:\n{history}"
        if history
        else question
    )
    result = run_fed_agent(
        full_question,
        context_text=source_text[:80_000],
        analysis_context=json.dumps(ml_context, indent=2, default=str),
    )
    return result.answer, result.tool_trace, True


def _render_result_agent_chat(panel_key: str, ml_context: dict[str, object], source_text: str) -> None:
    st.markdown("### Ask the AI analyst about these ML results")
    st.write(
        "Now that the model has produced a result, use the agent to explain what the prediction means, "
        "why rate relevance matters, and what limitations to mention. With a Portkey key, Claude can call project tools; "
        "without a key, a local tool-free analyst explanation is still available."
    )
    st.caption(
        "If you use the Claude button, your prompt, the selected model output, and the analyzed text are sent to NYU Portkey. "
        "Do not paste private or sensitive text here."
    )

    portkey_key = st.text_input(
        "Optional temporary Portkey API key for this explanation",
        type="password",
        key=f"{panel_key}_portkey_key",
        help="Leave blank to use PORTKEY_API_KEY from your shell. The key is not written to the repo or zip.",
    )
    if portkey_key.strip():
        os.environ["PORTKEY_API_KEY"] = portkey_key.strip()

    status = portkey_setup_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("Agent mode", "Claude + tools" if status["api_key_configured"] and status["portkey_ai_installed"] else "Local explanation")
    c2.metric("Portkey package", "Installed" if status["portkey_ai_installed"] else "Missing")
    c3.metric("API key", "Configured" if status["api_key_configured"] else "Not set")

    messages_key = f"{panel_key}_agent_messages"
    trace_key = f"{panel_key}_agent_trace"
    context_key = f"{panel_key}_agent_context_id"
    current_context_id = _context_id(ml_context, source_text)
    if st.session_state.get(context_key) != current_context_id:
        st.session_state[context_key] = current_context_id
        st.session_state[messages_key] = []
        st.session_state[trace_key] = []

    if st.button("Explain these ML results", key=f"{panel_key}_agent_explain", type="secondary"):
        question = (
            "Explain this ML result to a non-technical user. Focus on the prediction, confidence, "
            "rate relevance, early-stability, visible hawkish/dovish language, and limitations."
        )
        st.session_state[messages_key].append({"role": "user", "content": question})
        with st.spinner("Agent is reading the model output..."):
            try:
                answer, trace, used_portkey = _run_inline_agent(
                    question,
                    ml_context=ml_context,
                    source_text=source_text,
                    previous_messages=st.session_state[messages_key],
                )
            except AgentUnavailableError as exc:
                answer, trace, used_portkey = str(exc), [], False
            except Exception as exc:
                answer, trace, used_portkey = f"Agent call failed: {exc}", [], False
        st.session_state[messages_key].append({"role": "assistant", "content": answer})
        st.session_state[trace_key] = trace
        st.caption("Response source: Claude/Portkey" if used_portkey else "Response source: local explanation")

    followup = st.text_input(
        "Ask a follow-up about this specific prediction",
        key=f"{panel_key}_agent_followup",
        placeholder="Example: Why did it say no clear rate signal even though inflation appears in the text?",
    )
    if st.button("Ask follow-up", key=f"{panel_key}_agent_ask"):
        if not followup.strip():
            st.warning("Type a follow-up question first.")
        else:
            st.session_state[messages_key].append({"role": "user", "content": followup.strip()})
            with st.spinner("Agent is answering with this result as context..."):
                try:
                    answer, trace, used_portkey = _run_inline_agent(
                        followup.strip(),
                        ml_context=ml_context,
                        source_text=source_text,
                        previous_messages=st.session_state[messages_key],
                    )
                except AgentUnavailableError as exc:
                    answer, trace, used_portkey = str(exc), [], False
                except Exception as exc:
                    answer, trace, used_portkey = f"Agent call failed: {exc}", [], False
            st.session_state[messages_key].append({"role": "assistant", "content": answer})
            st.session_state[trace_key] = trace
            st.caption("Response source: Claude/Portkey" if used_portkey else "Response source: local explanation")

    for msg in st.session_state.get(messages_key, []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.get(trace_key):
        with st.expander("Project tools the agent called", expanded=False):
            st.json(st.session_state[trace_key])


with tab_data:
    st.caption(f"DATA_DIR: `{DATA_DIR}`")
    try:
        fomc = load_fomc()
        speaker = load_speaker()
        combined = build_combined_eda_frame(fomc, speaker)
        fomc_d, speaker_d = add_parsed_dates(fomc, speaker)
        st.success(f"Loaded FOMC: {len(fomc)} rows, Speaker: {len(speaker)} rows")
        st.dataframe(combined.head(20), use_container_width=True)

        work = combined.assign(
            date_parsed=pd.to_datetime(
                combined["date"].astype(str).str.replace(r"\.0$", "", regex=True),
                format="%Y%m%d",
                errors="coerce",
            )
        ).dropna(subset=["date_parsed"])
        if work.empty:
            st.warning("No valid YYYYMMDD dates for time series plot.")
        else:
            ts = (
                work.groupby([pd.Grouper(key="date_parsed", freq="ME"), "decision"])
                .size()
                .unstack(fill_value=0)
            )
            st.subheader("Documents per month by decision")
            fig, ax = plt.subplots(figsize=(10, 4))
            ts.plot(ax=ax, marker=".")
            ax.set_xlabel("Month")
            ax.set_ylabel("Count")
            ax.legend(title="decision")
            st.pyplot(fig)
            plt.close(fig)

            fred = _load_fred_history("FEDFUNDS")
            if not fred.empty:
                st.subheader("FRED effective federal funds rate")
                st.line_chart(fred.set_index("obs_date")["value"])
                with st.expander("Rate history with model predictions overlaid", expanded=True):
                    st.write(
                        "This is the quickest visual explanation of the project: the black line is the actual Fed funds rate, "
                        "and the dots show where the model's historical rate-direction call matched or missed the mapped outcome."
                    )
                    _render_fred_prediction_overlay(load_policy_artifact())
    except Exception as e:
        st.error(str(e))

with tab_policy:
    st.subheader("Speech early-warning system")
    st.write(
        "Paste Fed text or select a corpus speech. The app simulates reading from the beginning and updates "
        "whether the text looks like a signal for a rate cut, hold, hike, or no clear rate call. "
        "The pasted-text path only needs the included tuned model artifact."
    )
    artifact = load_policy_artifact()
    latest_dir = ROOT / "artifacts" / "policy_signal_tuning" / "latest"
    if artifact is None:
        st.warning(
            "No tuned policy-signal artifact found yet. Run "
            "`python scripts/tune_policy_signal_models.py` first."
        )
    else:
        metrics = artifact.metrics or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Best saved model", str(metrics.get("family", "unknown")))
        c2.metric("Correct by halfway", f"{100 * float(metrics.get('online_early_by_50_rate', 0)):.0f}%")
        c3.metric("Correct before obvious cue", f"{100 * float(metrics.get('online_before_cue_rate', 0)):.0f}%")

        leaderboard_path = latest_dir / "leaderboard_sorted.csv"
        with st.expander("Under the hood: model statistics", expanded=False):
            summary_stats = {}
            summary_path = latest_dir / "tuning_summary.json"
            if summary_path.is_file():
                try:
                    summary_stats = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    summary_stats = {}
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Action macro-F1", f"{float(metrics.get('action_f1_macro', 0)):.3f}")
            m2.metric("Direction macro-F1", f"{float(metrics.get('direction_f1_macro', 0)):.3f}")
            m3.metric("Not-confident rate", f"{100 * float(metrics.get('uncertain_rate_signal_rate', 0)):.0f}%")
            claude_cache_path = ROOT / "artifacts" / "policy_signal_tuning" / "claude_policy_features.csv"
            claude_rows = 0
            if claude_cache_path.is_file():
                try:
                    claude_rows = len(pd.read_csv(claude_cache_path, usecols=["doc_hash"]))
                except Exception:
                    claude_rows = 0
            m4.metric("Claude-labeled rows", f"{claude_rows:,}")
            eval_labels = str(metrics.get("action_eval_labels", "no_rate_signal,lower,maintain,raise"))
            all_label_f1 = metrics.get("action_f1_macro_all_labels")
            label_words = ", ".join(_plain_label(label) for label in eval_labels.split(",") if label)
            if all_label_f1 is not None:
                st.caption(
                    f"Primary Action macro-F1 is computed over classes present in the held-out test set: {label_words}. "
                    f"All-label macro-F1, including missing classes, is {float(all_label_f1):.3f}."
                )
            dummy_f1 = summary_stats.get("dummy_most_frequent_action_f1")
            if dummy_f1 is not None:
                st.caption(
                    f"Most-frequent baseline macro-F1 is {float(dummy_f1):.3f} "
                    f"(always predicting {_plain_label(summary_stats.get('dummy_most_frequent_label', 'maintain'))})."
                )
            if "raise" not in {label.strip() for label in eval_labels.split(",")}:
                st.info(
                    "The current held-out test window has no actual hike examples, so hike recall cannot be judged from this split. "
                    "That is a data-regime limitation, not a UI bug."
                )
            st.info(
                "Current model context features include the target rate range, rate midpoint, FRED macro/rate context, "
                "meeting-proximity alignment, speaker tier, document quality flags, and any cached Claude teacher labels."
            )
            st.markdown(
                """
                - **Action macro-F1** balances performance across the outcomes that actually appear in the held-out test split.
                - **Correct by halfway** asks whether the model becomes stably correct before 50% of a held-out test speech.
                - **Before obvious cue** asks whether the model was right before direct phrases like "raise rates" or "maintain the target range."
                - These scores are historical diagnostics, not a guarantee about a future meeting.
                """
            )

        _render_featured_visuals_strip()

        st.markdown("### 1. Choose input")
        input_mode = st.radio(
            "Input mode",
            ["Paste your own text", "Select from corpus"],
            horizontal=True,
            help="Pasted text works with only the included model artifact. Corpus selection requires data/parquet/.",
        )

        if input_mode == "Paste your own text":
            custom_text = st.text_area(
                "Paste a Fed speech, statement, press conference excerpt, or hypothetical policy text",
                height=_calculate_text_area_height(st.session_state.get("policy_paste_text", ""), min_height=260),
                placeholder="Paste Fed communication text here...",
                key="policy_paste_text",
            )
            if st.button("Run early-warning model on pasted text", type="primary"):
                if not custom_text.strip():
                    st.warning("Paste some Fed-related text first.")
                else:
                    st.session_state["policy_latest_result"] = {
                        "mode": "paste",
                        "text": custom_text,
                        "speaker": "custom",
                        "target": None,
                        "context_rows": None,
                    }
            latest_policy = st.session_state.get("policy_latest_result")
            if latest_policy and latest_policy.get("mode") == "paste":
                _render_policy_timeline(
                    artifact,
                    str(latest_policy.get("text", "")),
                    speaker=str(latest_policy.get("speaker", "custom")),
                    target=latest_policy.get("target"),
                    context_rows=latest_policy.get("context_rows"),
                )

        else:
            try:
                speaker_df = load_speaker().copy()
                speaker_df["date_sort"] = pd.to_datetime(
                    speaker_df["date"].astype(str).str.replace(r"\.0$", "", regex=True),
                    format="%Y%m%d",
                    errors="coerce",
                )
                speaker_df = speaker_df.sort_values("date_sort", ascending=False).head(250).reset_index(drop=True)
                labels = (
                    speaker_df["date"].astype(str)
                    + " | "
                    + speaker_df["participant"].astype(str)
                    + " | "
                    + speaker_df["domain"].astype(str)
                    + " | "
                    + speaker_df["decision"].astype(str)
                )
                selected = st.selectbox(
                    "Recent Fed speech/interview",
                    options=list(range(len(speaker_df))),
                    format_func=lambda i: labels.iloc[i],
                )
                row = speaker_df.iloc[int(selected)]
                target = weak_action_label(str(row["document"]), row.get("decision", "unknown"), source="speaker")
                meta_cols = st.columns(4)
                meta_cols[0].metric("Historical outcome", _plain_label(target))
                meta_cols[1].metric("Words", f"{int(row.get('word_count', len(str(row['document']).split()))):,}")
                meta_cols[2].metric("Speaker", str(row.get("participant", "unknown"))[:28])
                meta_cols[3].metric("Domain", str(row.get("domain", "unknown")).replace("www.", "")[:28])
                with st.expander("Speech preview", expanded=True):
                    preview_height = _calculate_text_area_height(str(row["document"])[:2500], min_height=220, max_height=400)
                    st.text_area("First part of selected speech", str(row["document"])[:2500], height=preview_height, label_visibility="collapsed", key=f"corpus_preview_{selected}")

                if st.button("Run model on selected speech", type="primary"):
                    context = _context_for_speech_rows(pd.DataFrame([row]))
                    st.session_state["policy_latest_result"] = {
                        "mode": "corpus",
                        "text": str(row["document"]),
                        "speaker": str(row.get("participant", "")),
                        "target": target,
                        "context_rows": context,
                    }
                latest_policy = st.session_state.get("policy_latest_result")
                if latest_policy and latest_policy.get("mode") == "corpus":
                    _render_policy_timeline(
                        artifact,
                        str(latest_policy.get("text", "")),
                        speaker=str(latest_policy.get("speaker", "")),
                        target=latest_policy.get("target"),
                        context_rows=latest_policy.get("context_rows"),
                    )
            except Exception as e:
                st.warning(
                    "Corpus-based speech selection requires local `data/parquet/` files. "
                    "The pasted-text early-warning demo above still works with the included model artifact."
                )
                st.caption(str(e))

        with st.expander("Featured visual: macro forecast from recent corpus documents", expanded=True):
            st.write(
                "This view averages model signals across recent Fed speeches, statements, minutes, and press conferences. "
                "It is a demo forecast, not a calibrated trading signal."
            )
            n_docs = st.slider("Recent documents to aggregate", min_value=5, max_value=60, value=20, step=5)
            if st.button("Run macro forecast"):
                try:
                    speaker = load_speaker().copy()
                    fomc = load_fomc().copy()
                    speaker_part = speaker[["date", "participant", "domain", "document"]].copy()
                    speaker_part["source"] = "speaker"
                    fomc_part = fomc[["date", "document"]].copy()
                    fomc_part["participant"] = "FOMC"
                    fomc_part["domain"] = fomc.get("type", "fomc").astype(str) if "type" in fomc.columns else "fomc"
                    fomc_part["source"] = "fomc"
                    all_docs = pd.concat([speaker_part, fomc_part], ignore_index=True)
                    all_docs["date_sort"] = pd.to_datetime(
                        all_docs["date"].astype(str).str.replace(r"\.0$", "", regex=True),
                        format="%Y%m%d",
                        errors="coerce",
                    )
                    recent = (
                        all_docs.dropna(subset=["date_sort"])
                        .sort_values("date_sort", ascending=False)
                        .head(int(n_docs))
                        .reset_index(drop=True)
                    )
                    recent_texts = recent["document"].astype(str).tolist()
                    forecast_df = _macro_forecast_from_texts(
                        artifact,
                        recent_texts,
                        context_rows=_context_for_document_rows(recent),
                    )
                    if forecast_df.empty:
                        st.warning("No usable recent corpus documents were available for aggregation.")
                    else:
                        probs = forecast_df[["lower", "maintain", "raise"]].mean()
                        most_likely = str(probs.idxmax())
                        st.markdown(f"## Across recent Fed communications, the model leans: {_plain_label(most_likely)}")
                        _render_prediction_badge(most_likely, prefix="Macro forecast")
                        m1, m2 = st.columns(2)
                        m1.metric("Documents analyzed", len(forecast_df))
                        m2.metric("Average rate relevance", _confidence_band(forecast_df["rate_relevant"].mean()))
                        st.caption("This is a simple average of document-level model probabilities across recent communications.")
                        st.bar_chart(_display_probability_series(probs))
                        with st.expander("Aggregated document-level outputs", expanded=False):
                            display_df = forecast_df.copy()
                            display_df["prediction_plain_english"] = display_df["prediction"].map(_plain_label)
                            st.dataframe(display_df, use_container_width=True)
                except Exception as e:
                    st.warning(
                        "Macro forecast requires local corpus data. Pasted-text inference still works without it."
                    )
                    st.caption(str(e))

        with st.expander("Model comparison and validation charts", expanded=False):
            if leaderboard_path.is_file():
                leaderboard = pd.read_csv(leaderboard_path).head(12)
                st.markdown("**Which model did best?**")
                if {"family", "variant", "action_f1_macro"}.issubset(leaderboard.columns):
                    chart_df = leaderboard.assign(model=leaderboard["family"].astype(str) + " " + leaderboard["variant"].astype(str))
                    st.bar_chart(chart_df.set_index("model")["action_f1_macro"])
                with st.expander("Model leaderboard table", expanded=False):
                    st.dataframe(leaderboard, use_container_width=True)

            st.markdown("**Where does the model get confused?**")
            _render_plain_confusion_matrix(latest_dir)

            speed_path = latest_dir / "best_online_speed_predictions.csv"
            if speed_path.is_file():
                speed = pd.read_csv(speed_path)
                _render_temporal_f1(speed)
                _render_speaker_role_accuracy(speed)

            with st.expander("Original technical plots", expanded=False):
                chart_cols = st.columns(3)
                for col, image_name in zip(
                    chart_cols,
                    ["leaderboard_model_comparison.png", "accuracy_vs_speed.png", "best_action_confusion.png"],
                ):
                    image_path = latest_dir / image_name
                    if image_path.is_file():
                        col.image(str(image_path), use_container_width=True)

with tab_analyze:
    st.subheader("Analyze any Fed text")
    st.write("Paste a speech excerpt, statement, or hypothetical paragraph. The result starts with a plain-English verdict; technical details are hidden below.")
    text_input_key = "analyze_text_input"
    text = st.text_area(
        "Paste text",
        height=_calculate_text_area_height(st.session_state.get(text_input_key, ""), min_height=240),
        placeholder="FOMC or speech excerpt…",
        key=text_input_key,
    )
    _ANALYZE_CACHE = "analyze_tab_last_result"
    if st.button("Run local analysis"):
        if text.strip():
            with st.spinner("Analyzing…"):
                st.session_state[_ANALYZE_CACHE] = {"text": text.strip(), "out": analyze_text(text)}
        else:
            st.warning("Enter some text.")

    _cached = st.session_state.get(_ANALYZE_CACHE)
    _show_analyze = bool(
        _cached
        and _cached.get("text") == text.strip()
        and text.strip()
    )
    if _show_analyze:
        out = _cached["out"]
        st.subheader("Analysis result")
        probs = out.get("prediction_proba") or {}
        pred_label = out.get("prediction_decision") or "unavailable"
        confidence = out.get("prediction_confidence")
        if confidence is None:
            confidence = max(probs.values()) if isinstance(probs, dict) and probs else None
        tb = out.get("sentiment_textblob") or {}
        fin = out.get("sentiment_financial") or {}
        st.markdown(f"## {_label_info(pred_label)['sentence']}")
        _render_prediction_badge(pred_label)
        if confidence is not None:
            _render_confidence_gauge(confidence)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Word count", f"{int(out.get('word_count', len(text.split()))):,}")
        p2.metric("Policy/rate relevance", _confidence_band(out.get("prediction_rate_relevance", 0)))
        p3.metric("General tone", "Positive" if float(tb.get("polarity", 0)) > 0.05 else "Negative" if float(tb.get("polarity", 0)) < -0.05 else "Neutral")
        p4.metric("Financial tone", "Hawkish-ish" if float(fin.get("financial_sentiment_score", 0)) < -0.02 else "Dovish-ish" if float(fin.get("financial_sentiment_score", 0)) > 0.02 else "Mixed")
        if out.get("prediction_model_type"):
            st.caption(f"Model source: `{out.get('prediction_model_type')}`")

        no_signal_probability = probs.get("no_rate_signal") if isinstance(probs, dict) else None
        _render_rate_relevance_scanner(text, out.get("prediction_rate_relevance"), no_signal_probability)
        _render_explanation(pred_label, text, confidence or 0, out.get("prediction_rate_relevance"))

        # SHAP-style highlighted text preview
        top_features = out.get("explanation_top_features") or []
        if top_features:
            st.markdown("### 📝 Speech analysis with feature importance")
            st.caption("Words highlighted by their contribution to the model's prediction (brighter = stronger influence)")
            _render_shap_highlighted_text(text, top_features, max_words=600)

        if isinstance(probs, dict) and probs:
            st.markdown("### Probability breakdown")
            st.bar_chart(_display_probability_series(probs))

        st.markdown("### Hawkish vs. dovish language found")
        _render_tone_breakdown(text)

        top_features = out.get("explanation_top_features") or []
        if top_features:
            feat_df = pd.DataFrame(top_features)
            if {"token", "contribution"}.issubset(feat_df.columns):
                st.markdown("### Technical token contributions")
                st.bar_chart(feat_df.set_index("token")["contribution"])
            with st.expander("Top feature table", expanded=False):
                st.dataframe(feat_df, use_container_width=True)

        if out.get("summary_textrank"):
            st.markdown("**Extractive summary**")
            st.write(out.get("summary_textrank"))

        if out.get("wordcloud_png_base64"):
            st.markdown("### Word cloud")
            st.image(base64.b64decode(out["wordcloud_png_base64"]), use_container_width=True)
        with st.expander("Under the hood: raw API-style output", expanded=False):
            st.json(out)

        hawk, dove = _tone_counts(text)
        agent_out = {k: v for k, v in out.items() if k != "wordcloud_png_base64"}
        agent_out.update(
            {
                "result_type": "custom_text_analysis",
                "final_prediction": pred_label,
                "confidence": confidence or 0,
                "rate_relevance": out.get("prediction_rate_relevance", 0),
                "hawkish_phrase_hits": sum(hawk.values()),
                "dovish_phrase_hits": sum(dove.values()),
                "probabilities": probs,
            }
        )
        _render_result_agent_chat("analyze_result", agent_out, text)

with tab_api:
    base = st.text_input("Flask base URL", value="http://127.0.0.1:5000")
    if st.button("GET /api/v1/sample"):
        try:
            r = requests.get(f"{base.rstrip('/')}/api/v1/sample", timeout=30)
            st.json(r.json() if r.ok else {"status": r.status_code, "body": r.text})
        except Exception as e:
            st.error(str(e))
    st.divider()
    api_text_key = "api_text_input"
    api_text = st.text_area(
        "Text for API",
        height=_calculate_text_area_height(st.session_state.get(api_text_key, ""), min_height=120),
        key=api_text_key,
    )
    _API_CACHE = "api_tab_last_analyze"
    if st.button("POST /api/v1/analyze"):
        try:
            base_norm = base.rstrip("/")
            r = requests.post(
                f"{base_norm}/api/v1/analyze",
                json={"text": api_text},
                timeout=120,
            )
            payload = r.json() if r.ok else {"status": r.status_code, "body": r.text}
            if r.ok:
                st.session_state[_API_CACHE] = {
                    "base": base_norm,
                    "text": api_text,
                    "payload": payload,
                }
            elif _API_CACHE in st.session_state:
                del st.session_state[_API_CACHE]
            if not r.ok:
                st.json(payload)
        except Exception as e:
            st.error(str(e))
            if _API_CACHE in st.session_state:
                del st.session_state[_API_CACHE]

    _api_cached = st.session_state.get(_API_CACHE)
    _show_api = bool(
        _api_cached
        and _api_cached.get("text") == api_text
        and _api_cached.get("base") == base.rstrip("/")
    )
    if _show_api:
        payload = _api_cached["payload"]
        pred_label = payload.get("prediction_decision") or "unavailable"
        confidence = payload.get("prediction_confidence")
        st.markdown(f"## {_label_info(pred_label)['sentence']}")
        _render_prediction_badge(pred_label, prefix="API verdict")
        if confidence is not None:
            _render_confidence_gauge(confidence)
        no_signal_probability = (
            payload.get("prediction_proba", {}).get("no_rate_signal")
            if isinstance(payload.get("prediction_proba"), dict)
            else None
        )
        _render_rate_relevance_scanner(api_text, payload.get("prediction_rate_relevance"), no_signal_probability)
        
        # SHAP highlighting for API results
        api_top_features = payload.get("explanation_top_features") or []
        if api_top_features:
            st.markdown("### 📝 API text with feature importance")
            st.caption("Words highlighted by their contribution to the model's prediction")
            _render_shap_highlighted_text(api_text, api_top_features, max_words=600)
        
        if payload.get("prediction_proba"):
            st.bar_chart(_display_probability_series(payload["prediction_proba"]))
        hawk, dove = _tone_counts(api_text)
        agent_payload = {k: v for k, v in payload.items() if k != "wordcloud_png_base64"}
        agent_payload.update(
            {
                "result_type": "api_text_analysis",
                "final_prediction": pred_label,
                "confidence": confidence or 0,
                "rate_relevance": payload.get("prediction_rate_relevance", 0),
                "hawkish_phrase_hits": sum(hawk.values()),
                "dovish_phrase_hits": sum(dove.values()),
                "probabilities": payload.get("prediction_proba") or {},
            }
        )
        _render_result_agent_chat("api_result", agent_payload, api_text)
        with st.expander("Raw API response", expanded=False):
            st.json(payload)
