"""FRED daily target range (DFEDTARL / DFEDTARU) for labels when statements are missing or unparseable."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import requests

from scraping.fed_official import fmt_rate

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def _fetch_series(series_id: str, timeout: float = 90.0) -> pd.Series:
    url = FRED_CSV.format(sid=series_id)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["observation_date"])
    s = df.set_index("observation_date").iloc[:, 0].astype(float)
    s = s.sort_index()
    return s


def load_target_range_frame(*, timeout: float = 90.0) -> pd.DataFrame:
    """Columns: tarl (lower limit), taru (upper limit) — Fed target range."""
    lower = _fetch_series("DFEDTARL", timeout=timeout)
    upper = _fetch_series("DFEDTARU", timeout=timeout)
    df = pd.DataFrame({"tarl": lower, "taru": upper})
    df = df.sort_index()
    df = df.ffill()
    return df


def _row_asof(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    sub = df.loc[df.index <= ts]
    if sub.empty:
        return None
    return sub.iloc[-1]


def _row_strictly_before(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    sub = df.loc[df.index < ts]
    if sub.empty:
        return None
    return sub.iloc[-1]


def infer_policy_from_fred(bounds: pd.DataFrame, ymd: str) -> dict[str, Any]:
    """Map repo columns: high = lower bound of range, low = upper bound (see fed_official.parse_monetary_statement)."""
    dt = pd.Timestamp(ymd)
    before = _row_strictly_before(bounds, dt)
    after = _row_asof(bounds, dt)
    if after is None:
        return {"error": "no FRED data on or before meeting", "decision": "maintain", "high": "0", "low": "0"}
    high_a = float(after["tarl"])
    low_a = float(after["taru"])
    if before is None:
        decision = "maintain"
    else:
        lb, ub = float(before["tarl"]), float(before["taru"])
        if high_a > lb + 1e-6 and low_a > ub + 1e-6:
            decision = "raise"
        elif high_a < lb - 1e-6 and low_a < ub - 1e-6:
            decision = "lower"
        else:
            decision = "maintain"
    return {
        "decision": decision,
        "high": fmt_rate(high_a),
        "low": fmt_rate(low_a),
        "source": "fred",
    }
