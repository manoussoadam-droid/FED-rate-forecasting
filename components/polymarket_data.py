"""Polymarket API integration and simulated Fed decision probability data."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

# Fed statements are released ~2:00 PM US Eastern (ET observing EST/EDT).
_FOMC_ET = ZoneInfo("America/New_York")

# FOMC meeting schedule for 2026 (statement release dates / meeting end dates per Fed calendar)
FOMC_MEETINGS_2026 = [
    datetime(2026, 1, 28, 14, 0, tzinfo=_FOMC_ET),
    datetime(2026, 3, 18, 14, 0, tzinfo=_FOMC_ET),
    datetime(2026, 4, 29, 14, 0, tzinfo=_FOMC_ET),
    datetime(2026, 6, 17, 14, 0, tzinfo=_FOMC_ET),
    datetime(2026, 7, 29, 14, 0, tzinfo=_FOMC_ET),
    datetime(2026, 9, 16, 14, 0, tzinfo=_FOMC_ET),
    datetime(2026, 10, 28, 14, 0, tzinfo=_FOMC_ET),
    datetime(2026, 12, 9, 14, 0, tzinfo=_FOMC_ET),
]


def get_next_fomc_meeting() -> datetime:
    """Return the next FOMC statement datetime (2:00 PM America/New_York)."""
    now = datetime.now(_FOMC_ET)
    for meeting in FOMC_MEETINGS_2026:
        if meeting > now:
            return meeting
    # If no meetings left in 2026, return last meeting
    return FOMC_MEETINGS_2026[-1]


@st.cache_data(ttl=300)
def fetch_polymarket_current() -> dict[str, float]:
    """
    Fetch current Fed rate decision probabilities from Polymarket API.
    
    Returns dict with keys: 'no_change', '25bps_cut', '25bps_hike', '50plus_cut', '50plus_hike'
    Falls back to simulated data if API unavailable.
    """
    try:
        # Try to fetch from Polymarket Gamma API
        # Note: Polymarket uses event slugs like "fed-decision-in-june" or similar
        # This is a placeholder - actual implementation would need the correct slug
        response = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"tag": "fomc", "active": "true", "limit": 1},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                event = data[0]
                # Parse outcome prices from the event
                # This would need to be adapted to actual Polymarket response structure
                outcomes = event.get("outcomePrices", [])
                if len(outcomes) >= 5:
                    return {
                        "no_change": float(outcomes[0]) * 100,
                        "25bps_cut": float(outcomes[1]) * 100,
                        "25bps_hike": float(outcomes[2]) * 100,
                        "50plus_cut": float(outcomes[3]) * 100,
                        "50plus_hike": float(outcomes[4]) * 100,
                    }
    except Exception:
        pass
    
    # Fallback to current simulated probabilities (based on screenshot)
    return {
        "no_change": 96.0,
        "25bps_cut": 2.6,
        "25bps_hike": 1.3,
        "50plus_cut": 0.05,
        "50plus_hike": 0.05,
    }


def _generate_realistic_series(
    start_val: float,
    end_val: float,
    num_points: int,
    volatility: float = 0.05,
    spike_day: int | None = None,
    spike_magnitude: float = 0.0
) -> list[float]:
    """Generate a realistic probability time series with gradual drift and optional spike."""
    series = []
    current = start_val
    drift_per_point = (end_val - start_val) / num_points
    
    for i in range(num_points):
        # Add gradual drift toward end value
        current += drift_per_point
        
        # Add random walk volatility
        noise = random.gauss(0, volatility * current)
        current += noise
        
        # Add spike if specified
        if spike_day and abs(i - spike_day) < 2:
            spike_factor = 1.0 - abs(i - spike_day) / 2.0
            current += spike_magnitude * spike_factor
        
        # Clamp to valid probability range
        current = max(0.1, min(99.9, current))
        series.append(current)
    
    return series


@st.cache_data(ttl=300)
def fetch_polymarket_historical_april() -> pd.DataFrame:
    """
    Generate historical Fed rate decision probabilities for April 2026.
    
    Returns DataFrame with columns:
    - date: datetime
    - outcome: str (no_change, 25bps_cut, 25bps_hike, 50plus_cut, 50plus_hike)
    - probability_polymarket: float
    - probability_our_model: float
    """
    # April 2026: 30 days
    start_date = datetime(2026, 4, 1, 0, 0)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    num_points = len(dates)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define probability evolution for April 2026
    # Based on plan: Start with uncertainty, converge to high No Change by end
    
    # Polymarket series (baseline)
    poly_no_change = _generate_realistic_series(
        start_val=45.0, end_val=96.0, num_points=num_points,
        volatility=0.08, spike_day=15, spike_magnitude=-8.0
    )
    poly_25bps_cut = _generate_realistic_series(
        start_val=35.0, end_val=2.6, num_points=num_points,
        volatility=0.06, spike_day=15, spike_magnitude=6.0
    )
    poly_25bps_hike = _generate_realistic_series(
        start_val=15.0, end_val=1.3, num_points=num_points,
        volatility=0.04, spike_day=15, spike_magnitude=3.0
    )
    poly_50plus_cut = _generate_realistic_series(
        start_val=3.0, end_val=0.05, num_points=num_points,
        volatility=0.02
    )
    poly_50plus_hike = _generate_realistic_series(
        start_val=2.0, end_val=0.05, num_points=num_points,
        volatility=0.02
    )
    
    # Normalize Polymarket probabilities to sum to 100
    poly_matrix = [
        [poly_no_change[i], poly_25bps_cut[i], poly_25bps_hike[i], 
         poly_50plus_cut[i], poly_50plus_hike[i]]
        for i in range(num_points)
    ]
    poly_normalized = []
    for row in poly_matrix:
        total = sum(row)
        poly_normalized.append([val / total * 100 for val in row])
    
    # Our model predictions: lag Polymarket by 1-2 days, add independent noise
    # Model should track well but not perfectly (correlation ~0.85-0.90)
    model_no_change = []
    model_25bps_cut = []
    model_25bps_hike = []
    model_50plus_cut = []
    model_50plus_hike = []
    
    for i in range(num_points):
        # Look back 1-2 days for lag effect
        lookback_idx = max(0, i - random.choice([1, 1, 2]))
        
        # Start with lagged Polymarket value
        lag_vals = poly_normalized[lookback_idx]
        
        # Add independent model divergence (5-10 percentage points possible)
        model_vals = []
        for val in lag_vals:
            divergence = random.gauss(0, 3.5)  # SD of 3.5% allows occasional 7-10% divergence
            model_val = val + divergence
            model_vals.append(max(0.01, model_val))
        
        # Normalize our model to sum to 100
        total = sum(model_vals)
        model_vals = [v / total * 100 for v in model_vals]
        
        model_no_change.append(model_vals[0])
        model_25bps_cut.append(model_vals[1])
        model_25bps_hike.append(model_vals[2])
        model_50plus_cut.append(model_vals[3])
        model_50plus_hike.append(model_vals[4])
    
    # Extract normalized Polymarket values
    poly_no_change_norm = [row[0] for row in poly_normalized]
    poly_25bps_cut_norm = [row[1] for row in poly_normalized]
    poly_25bps_hike_norm = [row[2] for row in poly_normalized]
    poly_50plus_cut_norm = [row[3] for row in poly_normalized]
    poly_50plus_hike_norm = [row[4] for row in poly_normalized]
    
    # Build DataFrame in long format
    records = []
    for i, date in enumerate(dates):
        records.append({
            "date": date,
            "outcome": "no_change",
            "probability_polymarket": poly_no_change_norm[i],
            "probability_our_model": model_no_change[i],
        })
        records.append({
            "date": date,
            "outcome": "25bps_cut",
            "probability_polymarket": poly_25bps_cut_norm[i],
            "probability_our_model": model_25bps_cut[i],
        })
        records.append({
            "date": date,
            "outcome": "25bps_hike",
            "probability_polymarket": poly_25bps_hike_norm[i],
            "probability_our_model": model_25bps_hike[i],
        })
        records.append({
            "date": date,
            "outcome": "50plus_cut",
            "probability_polymarket": poly_50plus_cut_norm[i],
            "probability_our_model": model_50plus_cut[i],
        })
        records.append({
            "date": date,
            "outcome": "50plus_hike",
            "probability_polymarket": poly_50plus_hike_norm[i],
            "probability_our_model": model_50plus_hike[i],
        })
    
    return pd.DataFrame(records)
