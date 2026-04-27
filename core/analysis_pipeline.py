"""Orchestrate text analysis components into one JSON-ready dict."""

from __future__ import annotations

import base64
import html
import io
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

from core.config import MAX_TEXT_CHARS_FOR_VECTOR, MAX_WORDCLOUD_WORDS
from core.predict import predict_decision
from core.sentiment_lm import score_lm_style
from core.summarize import summarize_openai, summarize_textrank
from core.text_clean import clean_for_ml

HAWKISH_PHRASES = (
    "higher for longer",
    "upside inflation risks",
    "persistent inflation",
    "inflation remains elevated",
    "further tightening",
    "additional tightening",
    "restrictive policy",
    "policy restraint",
    "strong labor market",
    "tight labor market",
    "rate hike",
    "raise rates",
    "inflation pressures",
    "price pressures",
    "hawkish",
)

DOVISH_PHRASES = (
    "disinflation progress",
    "inflation has eased",
    "slowing growth",
    "labor market softening",
    "downside risks",
    "economic slowdown",
    "policy easing",
    "rate cut",
    "lower rates",
    "accommodative policy",
    "dovish",
    "weaker demand",
    "recession risk",
    "softening labor market",
)

MACRO_BUCKETS = {
    "inflation": (
        "inflation",
        "prices",
        "price stability",
        "disinflation",
        "price pressures",
    ),
    "labor_market": (
        "labor market",
        "employment",
        "jobs",
        "wages",
        "unemployment",
    ),
    "growth": (
        "growth",
        "activity",
        "demand",
        "consumption",
        "investment",
        "slowdown",
        "recession",
    ),
    "policy": (
        "policy",
        "rate",
        "rates",
        "federal funds",
        "tightening",
        "easing",
        "restrictive",
        "accommodative",
    ),
}


def _sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", text)
    return max(1, len([p for p in parts if p.strip()]))


def _wordcloud_b64(text: str) -> str | None:
    t = clean_for_ml(text, max_chars=80_000)
    if len(t.split()) < 15:
        return None
    try:
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=MAX_WORDCLOUD_WORDS,
            collocations=False,
        ).generate(t)
        buf = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception:
        return None


def _count_phrase_hits(text: str, phrases: tuple[str, ...]) -> dict[str, int]:
    lower = text.lower()
    return {
        phrase: len(re.findall(rf"\b{re.escape(phrase.lower())}\b", lower))
        for phrase in phrases
        if re.search(rf"\b{re.escape(phrase.lower())}\b", lower)
    }


def _count_bucket_hits(text: str, phrases: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(phrase.lower())}\b", lower)) for phrase in phrases)


def _label_intensity(count: int) -> str:
    if count >= 8:
        return "high"
    if count >= 4:
        return "moderate"
    return "low"


def _label_presence(count: int) -> str:
    if count >= 8:
        return "dominant"
    if count >= 4:
        return "moderate"
    return "low"


def _tone_label(polarity: float, fin_score: float) -> str:
    blended = (polarity * 0.6) + (fin_score * 0.4)
    if blended >= 0.2:
        return "slightly positive"
    if blended <= -0.25:
        return "cautious"
    return "neutral"


def _extract_signal_excerpts(text: str, phrases: tuple[str, ...], color: str) -> list[dict[str, str]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    ranked: list[tuple[int, str, list[str]]] = []
    for sentence in sentences:
        lower = sentence.lower()
        matched = [phrase for phrase in phrases if re.search(rf"\b{re.escape(phrase.lower())}\b", lower)]
        if matched:
            ranked.append((len(matched), sentence, matched))
    ranked.sort(key=lambda item: item[0], reverse=True)

    excerpts: list[dict[str, str]] = []
    seen_sentences: set[str] = set()
    for _, sentence, matched in ranked:
        if sentence in seen_sentences:
            continue
        highlighted = html.escape(sentence)
        for phrase in sorted(matched, key=len, reverse=True):
            pattern = re.compile(re.escape(html.escape(phrase)), re.IGNORECASE)
            highlighted = pattern.sub(
                lambda m: (
                    f"<span style='background-color:{color}; color:#111827; "
                    "padding:0.1rem 0.25rem; border-radius:0.25rem; font-weight:600;'>"
                    f"{m.group(0)}</span>"
                ),
                highlighted,
            )
        excerpts.append(
            {
                "phrase": matched[0],
                "excerpt_html": highlighted,
            }
        )
        seen_sentences.add(sentence)
        if len(excerpts) == 3:
            break
    return excerpts


def _fallback_policy_label(fin_score: float, hawkish_hits: int, dovish_hits: int) -> str:
    spread = hawkish_hits - dovish_hits
    if spread >= 2 or fin_score >= 0.25:
        return "Hawkish"
    if spread <= -2 or fin_score <= -0.25:
        return "Dovish"
    return "Neutral"


def _confidence_label(primary_score: float) -> str:
    if primary_score >= 0.75:
        return "High"
    if primary_score >= 0.55:
        return "Medium"
    return "Low"


def _build_final_classification(
    prediction: dict[str, Any],
    fin_score: float,
    hawkish_hits: int,
    dovish_hits: int,
) -> dict[str, str]:
    model_label = prediction.get("prediction")
    probs = prediction.get("probabilities") or {}

    if model_label and probs:
        top_prob = max(float(v) for v in probs.values())
        label_map = {"high": "Hawkish", "low": "Dovish", "hold": "Neutral"}
        label = label_map.get(str(model_label).lower(), str(model_label).title())
        confidence = _confidence_label(top_prob)
        justification = (
            f"The model leans {label.lower()} with {confidence.lower()} confidence, supported by "
            f"{hawkish_hits} hawkish and {dovish_hits} dovish language cues in the excerpt."
        )
        return {"label": label, "confidence": confidence, "justification": justification}

    label = _fallback_policy_label(fin_score, hawkish_hits, dovish_hits)
    score = min(0.95, 0.5 + (abs(hawkish_hits - dovish_hits) * 0.08) + (abs(fin_score) * 0.35))
    confidence = _confidence_label(score)
    bias = "hawkish" if label == "Hawkish" else "dovish" if label == "Dovish" else "balanced"
    justification = (
        f"The text reads as {label.lower()} because the language mix shows a {bias} bias: "
        f"{hawkish_hits} hawkish signals versus {dovish_hits} dovish signals."
    )
    return {"label": label, "confidence": confidence, "justification": justification}


def _build_dashboard(text: str, polarity: float, fin: dict[str, Any]) -> dict[str, Any]:
    inflation_hits = _count_bucket_hits(text, MACRO_BUCKETS["inflation"])
    labor_hits = _count_bucket_hits(text, MACRO_BUCKETS["labor_market"])
    growth_hits = _count_bucket_hits(text, MACRO_BUCKETS["growth"])
    policy_hits = _count_bucket_hits(text, MACRO_BUCKETS["policy"])
    fin_score = float(fin.get("financial_sentiment_score") or 0.0)

    sentiment_bias = "hawkish bias" if fin_score >= 0.15 else "dovish bias" if fin_score <= -0.15 else "balanced bias"
    return {
        "macroeconomic_signal_overview": [
            f"Inflation mentions: {inflation_hits} ({_label_presence(inflation_hits)})",
            f"Labor market mentions: {labor_hits} ({_label_presence(labor_hits)})",
            f"Growth mentions: {growth_hits} ({_label_presence(growth_hits)})",
            f"Policy language intensity: {_label_intensity(policy_hits)}",
        ],
        "sentiment_indicators": [
            f"General tone: {_tone_label(polarity, fin_score)}",
            f"Financial sentiment score: {fin_score:.2f} ({sentiment_bias})",
            f"Positive vs negative signals: {int(fin.get('positive_hits', 0))} vs {int(fin.get('negative_hits', 0))}",
        ],
    }


def analyze_text(text: str) -> dict[str, Any]:
    raw = text or ""
    tb = TextBlob(raw)
    polarity = float(tb.sentiment.polarity)
    subjectivity = float(tb.sentiment.subjectivity)

    textrank = summarize_textrank(raw, sentences_count=3)
    oa = summarize_openai(raw)

    fin = score_lm_style(raw)
    pred = predict_decision(clean_for_ml(raw, MAX_TEXT_CHARS_FOR_VECTOR))
    hawkish_counts = _count_phrase_hits(raw, HAWKISH_PHRASES)
    dovish_counts = _count_phrase_hits(raw, DOVISH_PHRASES)
    hawkish_total = sum(hawkish_counts.values())
    dovish_total = sum(dovish_counts.values())
    signal_highlights = {
        "hawkish": _extract_signal_excerpts(raw, HAWKISH_PHRASES, "#fecaca"),
        "dovish": _extract_signal_excerpts(raw, DOVISH_PHRASES, "#bfdbfe"),
    }
    dashboard = _build_dashboard(raw, polarity, fin)
    final_classification = _build_final_classification(
        pred,
        float(fin.get("financial_sentiment_score") or 0.0),
        hawkish_total,
        dovish_total,
    )

    return {
        "word_count": len(raw.split()),
        "sentence_count": _sentence_count(raw),
        "sentiment_textblob": {
            "polarity": round(polarity, 4),
            "subjectivity": round(subjectivity, 4),
        },
        "sentiment_financial": fin,
        "summary_textrank": textrank,
        "summary_openai": oa.get("summary"),
        "summary_openai_error": oa.get("error"),
        "prediction_decision": pred.get("prediction"),
        "prediction_proba": pred.get("probabilities"),
        "explanation_top_features": pred.get("top_features"),
        "prediction_error": pred.get("error"),
        "metrics_dashboard": dashboard,
        "signal_highlights": signal_highlights,
        "signal_counts": {
            "hawkish": hawkish_counts,
            "dovish": dovish_counts,
        },
        "final_classification": final_classification,
        "wordcloud_png_base64": _wordcloud_b64(raw),
    }
