"""Orchestrate text analysis components into one JSON-ready dict."""

from __future__ import annotations

import base64
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


def analyze_text(text: str) -> dict[str, Any]:
    raw = text or ""
    tb = TextBlob(raw)
    polarity = float(tb.sentiment.polarity)
    subjectivity = float(tb.sentiment.subjectivity)

    textrank = summarize_textrank(raw, sentences_count=3)
    oa = summarize_openai(raw)

    fin = score_lm_style(raw)
    pred = predict_decision(clean_for_ml(raw, MAX_TEXT_CHARS_FOR_VECTOR))

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
        "wordcloud_png_base64": _wordcloud_b64(raw),
    }
