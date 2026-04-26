"""FRED data access layer.

Priority:
1. If FRED_API_KEY is set → use the ``fredapi`` library (proper REST API).
2. Otherwise → fall back to the public CSV download used by rebuild/fred_rates.py.

Results are cached in the ``fred_cache`` table of ``data/audit.sqlite`` to avoid
re-fetching unchanged series on every scheduler run.
"""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from core.config import AUDIT_DB, FRED_API_KEY

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
_FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_via_csv(series_id: str, timeout: float = 60.0) -> pd.DataFrame:
    """Public CSV endpoint — no API key required."""
    url = FRED_CSV_URL.format(sid=series_id)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "obs_date", df.columns[-1]: "value"})
    df["obs_date"] = df["obs_date"].dt.strftime("%Y-%m-%d")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["obs_date", "value"]].dropna(subset=["value"])


def _fetch_via_api(
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """FRED REST API — requires FRED_API_KEY."""
    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date
    resp = requests.get(_FRED_API_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    observations = resp.json().get("observations", [])
    rows = [
        {"obs_date": o["date"], "value": float(o["value"])}
        for o in observations
        if o.get("value") not in (".", None, "")
    ]
    return pd.DataFrame(rows, columns=["obs_date", "value"])


# ---------------------------------------------------------------------------
# SQLite cache helpers
# ---------------------------------------------------------------------------


def _db_conn() -> sqlite3.Connection:
    AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(AUDIT_DB)


def _cache_upsert(conn: sqlite3.Connection, series_id: str, df: pd.DataFrame) -> int:
    """Insert or replace rows in fred_cache; returns number of rows written."""
    now = _now_iso()
    rows = [
        (series_id, row["obs_date"], row["value"], now)
        for _, row in df.iterrows()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO fred_cache (series_id, obs_date, value, fetched_at) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def _cache_read(
    conn: sqlite3.Connection,
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    sql = "SELECT obs_date, value FROM fred_cache WHERE series_id = ?"
    params: list[Any] = [series_id]
    if start_date:
        sql += " AND obs_date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND obs_date <= ?"
        params.append(end_date)
    sql += " ORDER BY obs_date"
    rows = conn.execute(sql, params).fetchall()
    return pd.DataFrame(rows, columns=["obs_date", "value"])


def _cache_has_series(conn: sqlite3.Connection, series_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM fred_cache WHERE series_id = ? LIMIT 1", (series_id,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_series(
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    force_refresh: bool = False,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Return a DataFrame with columns ``obs_date`` (str) and ``value`` (float).

    Checks the ``fred_cache`` SQLite table first unless ``force_refresh=True``.
    Falls back to CSV download when ``FRED_API_KEY`` is not set.
    """
    conn = _db_conn()
    try:
        if not force_refresh and _cache_has_series(conn, series_id):
            cached = _cache_read(conn, series_id, start_date, end_date)
            if not cached.empty:
                return cached

        if FRED_API_KEY:
            df = _fetch_via_api(series_id, start_date, end_date, timeout=timeout)
        else:
            df = _fetch_via_csv(series_id, timeout=timeout)
            if start_date:
                df = df[df["obs_date"] >= start_date]
            if end_date:
                df = df[df["obs_date"] <= end_date]

        if not df.empty:
            _cache_upsert(conn, series_id, df)

        return df
    finally:
        conn.close()


def refresh_series_list(
    series_ids: list[str],
    timeout: float = 60.0,
) -> dict[str, int]:
    """Fetch and cache a list of FRED series. Returns {series_id: rows_written}."""
    results: dict[str, int] = {}
    conn = _db_conn()
    try:
        for sid in series_ids:
            try:
                if FRED_API_KEY:
                    df = _fetch_via_api(sid, timeout=timeout)
                else:
                    df = _fetch_via_csv(sid, timeout=timeout)
                n = _cache_upsert(conn, sid, df)
                results[sid] = n
            except Exception as exc:  # noqa: BLE001
                results[sid] = -1
                print(f"[fred_api] Failed to refresh {sid}: {exc}")
    finally:
        conn.close()
    return results


def series_to_dict(
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper returning a list of {obs_date, value} dicts."""
    df = fetch_series(series_id, start_date, end_date)
    return df.to_dict(orient="records")
