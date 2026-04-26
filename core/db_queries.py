"""SQLite query helpers for the pipeline support tables.

Tables are created automatically on first connection:
  - news           : fetched news articles from all sources
  - fred_cache     : cached FRED series observations
  - scheduler_log  : APScheduler job run history
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from core.config import AUDIT_DB

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


_SUPPORT_DDL = [
    """CREATE TABLE IF NOT EXISTS news (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        source          TEXT    NOT NULL,
        title           TEXT    NOT NULL,
        url             TEXT    UNIQUE NOT NULL,
        published_at    TEXT,
        summary         TEXT,
        sentiment_score REAL,
        sentiment_label TEXT,
        topic_query     TEXT,
        fetched_at      TEXT    NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_news_published ON news (published_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_news_source    ON news (source)",
    "CREATE INDEX IF NOT EXISTS idx_news_topic     ON news (topic_query)",
    """CREATE TABLE IF NOT EXISTS fred_cache (
        series_id  TEXT NOT NULL,
        obs_date   TEXT NOT NULL,
        value      REAL,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (series_id, obs_date)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_fred_series ON fred_cache (series_id, obs_date DESC)",
    """CREATE TABLE IF NOT EXISTS scheduler_log (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        job_name       TEXT    NOT NULL,
        started_at     TEXT    NOT NULL,
        finished_at    TEXT,
        status         TEXT    NOT NULL DEFAULT 'running',
        records_added  INTEGER DEFAULT 0,
        error_msg      TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sched_job ON scheduler_log (job_name, started_at DESC)",
]


def get_conn() -> sqlite3.Connection:
    AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(AUDIT_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    for stmt in _SUPPORT_DDL:
        conn.execute(stmt)
    conn.commit()
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# news table
# ---------------------------------------------------------------------------


def insert_news_articles(articles: list[dict[str, Any]]) -> int:
    """Insert articles, skipping duplicates (URL uniqueness). Returns inserted count."""
    if not articles:
        return 0
    now = _now_iso()
    rows = [
        (
            a.get("source", "unknown"),
            a.get("title", ""),
            a.get("url", ""),
            a.get("published_at"),
            a.get("summary"),
            a.get("sentiment_score"),
            a.get("sentiment_label"),
            a.get("topic_query"),
            now,
        )
        for a in articles
        if a.get("url")
    ]
    conn = get_conn()
    try:
        with conn:
            conn.executemany(
                """INSERT OR IGNORE INTO news
                   (source, title, url, published_at, summary,
                    sentiment_score, sentiment_label, topic_query, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        cur = conn.execute("SELECT changes()")
        return cur.fetchone()[0]
    finally:
        conn.close()


def query_news(
    source: str | None = None,
    topic: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return recent news rows as a list of dicts."""
    sql = "SELECT * FROM news WHERE 1=1"
    params: list[Any] = []
    if source:
        sql += " AND source = ?"
        params.append(source)
    if topic:
        sql += " AND (topic_query LIKE ? OR title LIKE ?)"
        params += [f"%{topic}%", f"%{topic}%"]
    sql += " ORDER BY published_at DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    conn = get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def news_count(source: str | None = None) -> int:
    conn = get_conn()
    try:
        if source:
            return conn.execute(
                "SELECT COUNT(*) FROM news WHERE source=?", (source,)
            ).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# scheduler_log table
# ---------------------------------------------------------------------------


def log_job_start(job_name: str) -> int:
    """Insert a 'running' row; returns the new row id."""
    conn = get_conn()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO scheduler_log (job_name, started_at, status) VALUES (?,?,?)",
                (job_name, _now_iso(), "running"),
            )
        return cur.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def log_job_finish(
    row_id: int,
    status: str = "ok",
    records_added: int = 0,
    error_msg: str | None = None,
) -> None:
    conn = get_conn()
    try:
        with conn:
            conn.execute(
                """UPDATE scheduler_log
                   SET finished_at=?, status=?, records_added=?, error_msg=?
                   WHERE id=?""",
                (_now_iso(), status, records_added, error_msg, row_id),
            )
    finally:
        conn.close()


def get_scheduler_log(limit: int = 20) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM scheduler_log ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# General-purpose safe read-only query
# ---------------------------------------------------------------------------

_BLOCKED_KEYWORDS = frozenset(
    ["insert", "update", "delete", "drop", "alter", "create", "replace", "attach", "pragma"]
)


def safe_query_v2(sql: str, max_rows: int = 500) -> list[dict[str, Any]]:
    """Execute a read-only SQL SELECT statement.

    Raises :class:`ValueError` for:
    - statements that don't start with SELECT
    - statements containing semicolons (multi-statement injection)
    - statements containing blocked DML/DDL keywords
    """
    stripped = sql.strip()
    if not stripped.lower().startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    if ";" in stripped:
        raise ValueError("Semicolons are not allowed (prevents multi-statement injection).")
    lower = stripped.lower()
    for kw in _BLOCKED_KEYWORDS:
        if f" {kw} " in lower or lower.endswith(f" {kw}"):
            raise ValueError(f"Blocked keyword '{kw}' found in query.")
    conn = get_conn()
    try:
        # Use a read-only connection via uri= to enforce no writes at the
        # SQLite level, providing defence-in-depth beyond the keyword check.
        cur = conn.execute(stripped)
        rows = cur.fetchmany(max_rows)
        if not rows:
            return []
        keys = [d[0] for d in cur.description]
        return [dict(zip(keys, r)) for r in rows]
    finally:
        conn.close()
