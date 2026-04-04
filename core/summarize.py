"""Extractive TextRank (sumy) + optional OpenAI abstractive summary."""

from __future__ import annotations

from typing import Any

from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

from core.config import OPENAI_API_KEY


def summarize_textrank(text: str, sentences_count: int = 3) -> str | None:
    if not text or len(text.strip()) < 80:
        return None
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        stemmer = Stemmer("english")
        summarizer = TextRankSummarizer(stemmer)
        summarizer.stop_words = getattr(summarizer, "stop_words", set())
        sentences = summarizer(parser.document, sentences_count)
        if not sentences:
            return None
        return " ".join(s._text for s in sentences)
    except Exception:
        return None


def summarize_openai(text: str, max_words: int = 200) -> dict[str, Any]:
    """Returns {summary: str|None, error: str|None}."""
    if not OPENAI_API_KEY:
        return {"summary": None, "error": "OPENAI_API_KEY not set"}
    if not text.strip():
        return {"summary": None, "error": "empty text"}
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "Summarize the following Federal Reserve style communication in plain English "
            f"in at most {max_words} words. Focus on policy stance and outlook.\n\n---\n"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a careful financial policy assistant."},
                {"role": "user", "content": prompt + text[:12000]},
            ],
            max_tokens=400,
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        return {"summary": (content or "").strip() or None, "error": None}
    except Exception as e:
        return {"summary": None, "error": str(e)}
