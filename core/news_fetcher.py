"""Financial news fetchers: Alpha Vantage (primary) and NewsAPI.org (secondary).

Both sources degrade gracefully when their API key is absent — they return an
empty list with a warning instead of raising.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from core.config import ALPHA_VANTAGE_KEY, NEWS_API_KEY

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    _news_retry = retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
except ImportError:
    def _news_retry(fn):  # type: ignore[misc]
        return fn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DEFAULT_TOPICS = [
    "federal reserve",
    "FOMC",
    "interest rates",
    "monetary policy",
    "inflation",
]

_FED_AV_TOPICS = "economy_fiscal,economy_monetary,finance"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_ago_iso(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y%m%dT%H%M")


# ---------------------------------------------------------------------------
# Alpha Vantage NEWS_SENTIMENT
# ---------------------------------------------------------------------------
# Docs: https://www.alphavantage.co/documentation/#news-sentiment
# Free tier: 500 calls/day, 25 calls/minute

_AV_BASE = "https://www.alphavantage.co/query"


def _av_sentiment_label(score: float) -> str:
    if score >= 0.35:
        return "Bullish"
    if score <= -0.35:
        return "Bearish"
    return "Neutral"


def fetch_alpha_vantage(
    query: str = "federal reserve",
    days_back: int = 7,
    limit: int = 50,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetch news + sentiment from Alpha Vantage.

    Returns a list of article dicts ready for ``db_queries.insert_news_articles``.
    Returns [] if ALPHA_VANTAGE_KEY is not set.
    """
    if not ALPHA_VANTAGE_KEY:
        log.warning("[alpha_vantage] ALPHA_VANTAGE_KEY not set — skipping.")
        return []

    time_from = _days_ago_iso(days_back)
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "",
        "topics": _FED_AV_TOPICS,
        "time_from": time_from,
        "limit": min(limit, 200),
        "sort": "LATEST",
        "apikey": ALPHA_VANTAGE_KEY,
    }
    try:
        @_news_retry
        def _fetch() -> dict:
            r = requests.get(_AV_BASE, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()

        data = _fetch()
    except Exception as exc:
        log.error("[alpha_vantage] Request failed after retries: %s", exc)
        return []

    feed = data.get("feed", [])
    if not feed:
        log.info("[alpha_vantage] Empty feed (quota exhausted or no results).")
        return []

    articles: list[dict[str, Any]] = []
    for item in feed:
        title = item.get("title", "")
        url = item.get("url", "")
        if not url:
            continue
        sentiment_score: float = float(item.get("overall_sentiment_score", 0.0))
        sentiment_label: str = item.get(
            "overall_sentiment_label", _av_sentiment_label(sentiment_score)
        )
        articles.append(
            {
                "source": "alpha_vantage",
                "title": title,
                "url": url,
                "published_at": item.get("time_published", ""),
                "summary": item.get("summary", ""),
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "topic_query": query,
            }
        )
    log.info("[alpha_vantage] Fetched %d articles.", len(articles))
    return articles


# ---------------------------------------------------------------------------
# NewsAPI.org
# ---------------------------------------------------------------------------
# Docs: https://newsapi.org/docs/endpoints/everything
# Free developer plan: 100 requests/day, articles up to 1 month old.

_NEWSAPI_BASE = "https://newsapi.org/v2/everything"
_NEWSAPI_SOURCES = (
    "reuters,the-wall-street-journal,financial-times,bloomberg,"
    "cnbc,marketwatch,fortune,business-insider"
)


def fetch_newsapi(
    query: str = "federal reserve",
    days_back: int = 7,
    limit: int = 100,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetch articles from NewsAPI.org.

    Returns a list of article dicts ready for ``db_queries.insert_news_articles``.
    Returns [] if NEWS_API_KEY is not set.
    """
    if not NEWS_API_KEY:
        log.warning("[newsapi] NEWS_API_KEY not set — skipping.")
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime(
        "%Y-%m-%d"
    )
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "pageSize": min(limit, 100),
        "language": "en",
        "apiKey": NEWS_API_KEY,
    }
    try:
        @_news_retry
        def _fetch() -> dict:
            r = requests.get(_NEWSAPI_BASE, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()

        data = _fetch()
    except Exception as exc:
        log.error("[newsapi] Request failed after retries: %s", exc)
        return []

    raw_articles = data.get("articles", [])
    if data.get("status") != "ok":
        log.warning("[newsapi] API returned status=%s: %s", data.get("status"), data.get("message"))
        return []

    articles: list[dict[str, Any]] = []
    for item in raw_articles:
        url = item.get("url", "")
        if not url or url == "https://removed.com":
            continue
        source_name = (item.get("source") or {}).get("name", "newsapi")
        articles.append(
            {
                "source": f"newsapi/{source_name}",
                "title": item.get("title", ""),
                "url": url,
                "published_at": item.get("publishedAt", ""),
                "summary": item.get("description", ""),
                "sentiment_score": None,
                "sentiment_label": None,
                "topic_query": query,
            }
        )
    log.info("[newsapi] Fetched %d articles for query=%r.", len(articles), query)
    return articles


# ---------------------------------------------------------------------------
# Combined fetcher (used by scheduler and MCP tool)
# ---------------------------------------------------------------------------


def fetch_all_news(
    topics: list[str] | None = None,
    days_back: int = 7,
) -> list[dict[str, Any]]:
    """Fetch from all available sources (Alpha Vantage + NewsAPI).

    Deduplication is done at insert time via URL UNIQUE.
    """
    topics = topics or DEFAULT_TOPICS
    all_articles: list[dict[str, Any]] = []

    for topic in topics:
        all_articles.extend(fetch_alpha_vantage(query=topic, days_back=days_back))
        all_articles.extend(fetch_newsapi(query=topic, days_back=days_back))

    return all_articles
