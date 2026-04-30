#!/usr/bin/env python3
"""MCP stdio server — FOMC corpus + continuous pipeline tools.

12 tools:
  1.  analyze_fed_text          — full NLP pipeline on raw text
  2.  random_corpus_document     — random row from Parquet corpus
  3.  trigger_scrape_fed         — run the Fed scraper inline
  4.  trigger_scrape_news        — fetch news from all configured sources
  5.  query_database             — safe read-only SQL on audit.sqlite
  6.  get_latest_documents       — newest N rows from Parquet
  7.  search_corpus              — keyword search across document text
  8.  get_fred_series            — FRED economic series (cache-first)
  9.  get_fomc_calendar          — upcoming FOMC meeting dates
  10. get_corpus_stats           — counts by year / type / decision
  11. get_audit_report           — ingestion_log + document_audit summary
  12. get_scheduler_status       — last 20 scheduler_log rows

Run from project root with venv activated:
    python tools/mcp_server.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Install mcp: pip install mcp", file=sys.stderr)
    sys.exit(1)

from core.analysis_pipeline import analyze_text
from core.config import AUDIT_DB, FOMC_PARQUET_DIR, SPEAKER_PARQUET_DIR
from core.db_queries import get_scheduler_log, insert_news_articles, query_news, safe_query_v2
from core.fred_api import series_to_dict
from core.ingest import load_fomc, load_speaker, random_corpus_row

mcp = FastMCP("FOMC Intelligence Pipeline")

# ---------------------------------------------------------------------------
# 1. analyze_fed_text  (existing, unchanged signature)
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_fed_text(text: str) -> str:
    """Run the full NLP analysis pipeline on raw Fed text.

    Returns JSON with: decision prediction, sentiment scores (TextBlob + LM),
    extractive summary, word count, and (if OpenAI key set) abstractive summary.
    """
    return json.dumps(analyze_text(text), indent=2, default=str)


# ---------------------------------------------------------------------------
# 2. random_corpus_document  (updated: Parquet-first)
# ---------------------------------------------------------------------------


@mcp.tool()
def random_corpus_document() -> str:
    """Return one random document from the combined FOMC + speaker corpus.

    Reads from Parquet. Returns JSON.
    """
    fomc = load_fomc()
    speaker = load_speaker()
    return json.dumps(random_corpus_row(fomc, speaker), indent=2, default=str)


# ---------------------------------------------------------------------------
# 3. trigger_scrape_fed
# ---------------------------------------------------------------------------


@mcp.tool()
def trigger_scrape_fed(
    check_speeches: bool = True,
    check_fomc_docs: bool = True,
) -> str:
    """Trigger a live scrape of federalreserve.gov for new documents.

    Checks the speeches JSON feed and/or the FOMC calendar page.
    Returns a JSON summary: {checked, new_speeches, new_fomc_docs, errors}.
    """
    import hashlib
    from scraping.fed_official import FedHttpSession, fetch_speeches_metadata

    session = FedHttpSession()
    result: dict[str, Any] = {
        "checked": [],
        "new_speeches": 0,
        "new_fomc_docs": 0,
        "errors": [],
    }

    if check_speeches:
        result["checked"].append("speeches_json")
        try:
            speeches = fetch_speeches_metadata(session)
            feed_hash = hashlib.sha256(
                json.dumps(speeches, sort_keys=True).encode()
            ).hexdigest()[:16]
            result["speeches_feed_hash"] = feed_hash
            result["speeches_in_feed"] = len(speeches)
        except Exception as exc:
            result["errors"].append(f"speeches: {exc}")

    if check_fomc_docs:
        result["checked"].append("fomc_calendar")
        try:
            from scraping.fed_official import scrape_minutes_meeting_dates

            meeting_dates = scrape_minutes_meeting_dates(session)
            result["fomc_meeting_dates_found"] = len(meeting_dates)
            result["most_recent_meeting_dates"] = sorted(meeting_dates, reverse=True)[:5]
        except Exception as exc:
            result["errors"].append(f"fomc_calendar: {exc}")

    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# 4. trigger_scrape_news
# ---------------------------------------------------------------------------


@mcp.tool()
def trigger_scrape_news(
    topic: str = "federal reserve",
    days_back: int = 7,
) -> str:
    """Fetch financial news from Alpha Vantage and NewsAPI.

    Deduplicates by URL and inserts new articles into the ``news`` SQLite table.
    Returns JSON: {fetched, inserted, sources_used, errors}.

    Args:
        topic: Search query (e.g. "FOMC", "interest rates", "inflation").
        days_back: How many days back to look for articles.
    """
    from core.news_fetcher import fetch_alpha_vantage, fetch_newsapi

    articles: list[dict[str, Any]] = []
    errors: list[str] = []
    sources_used: list[str] = []

    try:
        av = fetch_alpha_vantage(query=topic, days_back=days_back)
        articles.extend(av)
        if av:
            sources_used.append("alpha_vantage")
    except Exception as exc:
        errors.append(f"alpha_vantage: {exc}")

    try:
        na = fetch_newsapi(query=topic, days_back=days_back)
        articles.extend(na)
        if na:
            sources_used.append("newsapi")
    except Exception as exc:
        errors.append(f"newsapi: {exc}")

    inserted = insert_news_articles(articles)
    return json.dumps(
        {
            "fetched": len(articles),
            "inserted": inserted,
            "sources_used": sources_used,
            "errors": errors,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# 5. query_database
# ---------------------------------------------------------------------------


@mcp.tool()
def query_database(sql: str) -> str:
    """Run a read-only SQL SELECT on data/audit.sqlite.

    Available tables:
      - ingestion_log, document_audit, fetch_errors  (core audit)
      - news           (fetched articles: source, title, url, published_at, sentiment_*)
      - fred_cache     (FRED series observations: series_id, obs_date, value)
      - scheduler_log  (job run history: job_name, status, records_added)

    DML statements (INSERT, UPDATE, DELETE, DROP, etc.) are rejected.
    Returns up to 500 rows as a JSON array.

    Example queries:
      SELECT * FROM news ORDER BY published_at DESC LIMIT 20
      SELECT series_id, COUNT(*) as n FROM fred_cache GROUP BY series_id
      SELECT job_name, status, records_added FROM scheduler_log ORDER BY started_at DESC LIMIT 10
    """
    try:
        rows = safe_query_v2(sql, max_rows=500)
        return json.dumps(rows, indent=2, default=str)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Query failed: {exc}"})


# ---------------------------------------------------------------------------
# 6. get_latest_documents
# ---------------------------------------------------------------------------


@mcp.tool()
def get_latest_documents(n: int = 10, corpus: str = "fomc") -> str:
    """Return the N most recent documents from the Parquet corpus.

    Args:
        n: Number of rows to return (max 100).
        corpus: "fomc" or "speaker".

    Returns JSON array of rows with date, document_type, decision, source_url, word_count.
    """
    n = min(max(1, n), 100)
    try:
        if corpus == "speaker":
            df = load_speaker()
            df = df.sort_values("date", ascending=False).head(n)
            cols = ["date", "participant", "domain", "decision", "high", "low", "word_count"]
        else:
            df = load_fomc()
            df = df.sort_values("date", ascending=False).head(n)
            cols = ["date", "decision", "high", "low", "word_count"]
            if "type" in df.columns:
                cols = ["date", "type"] + [c for c in cols if c != "date"]

        cols = [c for c in cols if c in df.columns]
        rows = df[cols].to_dict(orient="records")
        return json.dumps(rows, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# 7. search_corpus
# ---------------------------------------------------------------------------


@mcp.tool()
def search_corpus(query: str, limit: int = 20, corpus: str = "both") -> str:
    """Search document text across the FOMC / speaker corpus.

    Args:
        query: Keyword or phrase to search for (case-insensitive substring match).
        limit: Maximum results per corpus (max 50).
        corpus: "fomc", "speaker", or "both".

    Returns JSON array of matching rows with date, decision, and a text snippet.
    """
    limit = min(max(1, limit), 50)
    q_lower = query.lower()

    def _search(df: "Any", label: str) -> list[dict[str, Any]]:
        if df.empty or "document" not in df.columns:
            return []
        mask = df["document"].str.lower().str.contains(q_lower, regex=False, na=False)
        hits = df[mask].head(limit)
        results = []
        for _, row in hits.iterrows():
            text: str = str(row.get("document", ""))
            idx = text.lower().find(q_lower)
            snippet = text[max(0, idx - 80): idx + 160].strip() if idx >= 0 else text[:200]
            results.append(
                {
                    "corpus": label,
                    "date": str(row.get("date", "")),
                    "decision": str(row.get("decision", "")),
                    "participant": str(row.get("participant", "")) if "participant" in row else None,
                    "snippet": f"...{snippet}...",
                    "word_count": int(row.get("word_count", 0)),
                }
            )
        return results

    try:
        all_results: list[dict[str, Any]] = []
        if corpus in ("fomc", "both"):
            all_results.extend(_search(load_fomc(), "fomc"))
        if corpus in ("speaker", "both"):
            all_results.extend(_search(load_speaker(), "speaker"))
        return json.dumps(all_results, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# 8. get_fred_series
# ---------------------------------------------------------------------------


@mcp.tool()
def get_fred_series(
    series_id: str,
    start_date: str = "",
    end_date: str = "",
) -> str:
    """Return observations for a FRED economic series.

    Checks the local ``fred_cache`` SQLite table first; fetches live if missing.
    Falls back to public CSV download when FRED_API_KEY is not set.

    Common series:
      DFEDTARL  — Fed funds target range lower bound
      DFEDTARU  — Fed funds target range upper bound
      FEDFUNDS  — Effective federal funds rate
      T10Y2Y    — 10Y-2Y Treasury spread (recession indicator)
      UNRATE    — Unemployment rate
      CPIAUCSL  — CPI all urban consumers

    Args:
        series_id: FRED series identifier (e.g. "FEDFUNDS").
        start_date: ISO date string "YYYY-MM-DD" (optional).
        end_date:   ISO date string "YYYY-MM-DD" (optional).

    Returns JSON array of {obs_date, value}.
    """
    try:
        data = series_to_dict(
            series_id.upper().strip(),
            start_date or None,
            end_date or None,
        )
        return json.dumps({"series_id": series_id, "count": len(data), "data": data}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# 9. get_fomc_calendar
# ---------------------------------------------------------------------------


@mcp.tool()
def get_fomc_calendar() -> str:
    """Scrape upcoming (and recent past) FOMC meeting dates from federalreserve.gov.

    Returns JSON list of {date, description, is_projection} objects.
    """
    try:
        import re
        import requests
        from bs4 import BeautifulSoup

        url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        resp = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "fomc_tools/1.0 educational research"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        meetings: list[dict[str, Any]] = []
        # The page uses <div class="fomc-meeting"> blocks (as of 2024–2026)
        for div in soup.select("div.fomc-meeting, div.panel-default"):
            header = div.select_one(
                ".fomc-meeting__date, .panel-heading, h5, h4"
            )
            if not header:
                continue
            date_text = header.get_text(" ", strip=True)
            is_projection = bool(div.select_one(".fomc-meeting--projected, .text-danger"))
            meetings.append(
                {
                    "date_text": date_text,
                    "is_projection": is_projection,
                }
            )

        if not meetings:
            # Broad fallback: find all date-looking text blocks
            for tag in soup.find_all(["h5", "h4", "strong"]):
                t = tag.get_text(strip=True)
                if re.search(r"\b(January|February|March|April|May|June|July|August|"
                             r"September|October|November|December)\b", t):
                    meetings.append({"date_text": t, "is_projection": False})

        return json.dumps(
            {"source_url": url, "count": len(meetings), "meetings": meetings[:40]},
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# 10. get_corpus_stats
# ---------------------------------------------------------------------------


@mcp.tool()
def get_corpus_stats() -> str:
    """Return high-level statistics about the Parquet + SQLite corpus.

    Includes: row counts, year range, decision distribution, document type counts,
    news table counts, fred_cache series list, and scheduler run summary.
    """
    import sqlite3

    stats: dict[str, Any] = {}

    # --- Parquet stats ---
    try:
        fomc = load_fomc()
        stats["fomc"] = {
            "total_rows": len(fomc),
            "year_range": [str(fomc["date"].min()[:4]), str(fomc["date"].max()[:4])]
            if not fomc.empty else [],
            "decision_counts": fomc["decision"].value_counts().to_dict()
            if "decision" in fomc.columns else {},
        }
        if "type" in fomc.columns:
            stats["fomc"]["type_counts"] = fomc["type"].value_counts().to_dict()
    except Exception as exc:
        stats["fomc"] = {"error": str(exc)}

    try:
        speaker = load_speaker()
        stats["speaker"] = {
            "total_rows": len(speaker),
            "year_range": [str(speaker["date"].min()[:4]), str(speaker["date"].max()[:4])]
            if not speaker.empty else [],
            "decision_counts": speaker["decision"].value_counts().to_dict()
            if "decision" in speaker.columns else {},
            "unique_participants": speaker["participant"].nunique()
            if "participant" in speaker.columns else 0,
        }
    except Exception as exc:
        stats["speaker"] = {"error": str(exc)}

    # --- SQLite pipeline stats ---
    if AUDIT_DB.exists():
        conn = sqlite3.connect(AUDIT_DB)
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()]
            stats["sqlite_tables"] = tables

            if "news" in tables:
                stats["news"] = {
                    "total_articles": conn.execute("SELECT COUNT(*) FROM news").fetchone()[0],
                    "by_source": dict(conn.execute(
                        "SELECT source, COUNT(*) FROM news GROUP BY source ORDER BY COUNT(*) DESC"
                    ).fetchall()),
                }

            if "fred_cache" in tables:
                stats["fred_cache"] = {
                    "series_cached": [
                        r[0] for r in conn.execute(
                            "SELECT DISTINCT series_id FROM fred_cache ORDER BY series_id"
                        ).fetchall()
                    ]
                }

            if "scheduler_log" in tables:
                last = conn.execute(
                    "SELECT job_name, status, finished_at FROM scheduler_log "
                    "ORDER BY started_at DESC LIMIT 5"
                ).fetchall()
                stats["scheduler_recent"] = [
                    {"job": r[0], "status": r[1], "finished_at": r[2]} for r in last
                ]
        finally:
            conn.close()

    return json.dumps(stats, indent=2, default=str)


# ---------------------------------------------------------------------------
# 11. get_audit_report
# ---------------------------------------------------------------------------


@mcp.tool()
def get_audit_report(limit_runs: int = 5) -> str:
    """Return a summary from the audit SQLite DB (ingestion runs + document audit).

    Args:
        limit_runs: How many recent ingestion_log runs to include.

    Returns JSON with ingestion runs, quality flags summary, and fetch errors.
    """
    import sqlite3

    if not AUDIT_DB.exists():
        return json.dumps({"error": "audit.sqlite not found. Run rebuild_database.py first."})

    report: dict[str, Any] = {}
    conn = sqlite3.connect(AUDIT_DB)
    conn.row_factory = sqlite3.Row
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        if "ingestion_log" in tables:
            runs = conn.execute(
                "SELECT * FROM ingestion_log ORDER BY started_at DESC LIMIT ?",
                (limit_runs,),
            ).fetchall()
            report["ingestion_runs"] = [dict(r) for r in runs]

        if "document_audit" in tables:
            total = conn.execute("SELECT COUNT(*) FROM document_audit").fetchone()[0]
            report["document_audit_total"] = total

            # quality flag distribution
            flag_rows = conn.execute(
                "SELECT quality_flags FROM document_audit WHERE quality_flags IS NOT NULL "
                "AND quality_flags != '[]'"
            ).fetchall()
            from collections import Counter
            import ast
            flag_counter: Counter = Counter()
            for row in flag_rows:
                try:
                    flags = ast.literal_eval(row[0]) if isinstance(row[0], str) else row[0]
                    if isinstance(flags, list):
                        flag_counter.update(flags)
                except Exception:
                    pass
            report["quality_flag_counts"] = dict(flag_counter.most_common(10))

        if "fetch_errors" in tables:
            err_count = conn.execute("SELECT COUNT(*) FROM fetch_errors").fetchone()[0]
            recent_errors = conn.execute(
                "SELECT url, http_status, message, occurred_at FROM fetch_errors "
                "ORDER BY occurred_at DESC LIMIT 10"
            ).fetchall()
            report["fetch_errors"] = {
                "total": err_count,
                "recent": [dict(r) for r in recent_errors],
            }
    finally:
        conn.close()

    return json.dumps(report, indent=2, default=str)


# ---------------------------------------------------------------------------
# 12. get_scheduler_status
# ---------------------------------------------------------------------------


@mcp.tool()
def get_scheduler_status(limit: int = 20) -> str:
    """Return the last N scheduler job run records from scheduler_log.

    Each record shows: job_name, started_at, finished_at, status, records_added, error_msg.

    To start the scheduler: python tools/scheduler.py
    """
    try:
        rows = get_scheduler_log(limit=limit)
        return json.dumps(
            {"total_shown": len(rows), "rows": rows},
            indent=2,
            default=str,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
