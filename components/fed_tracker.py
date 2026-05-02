"""Polymarket-style Fed decision probability tracker component."""

from __future__ import annotations

from datetime import datetime, timedelta

import plotly.graph_objects as go
import streamlit as st

from components.polymarket_data import (
    fetch_polymarket_current,
    fetch_polymarket_historical_april,
    get_next_fomc_meeting,
)

# Custom CSS: make st.metric display large blue countdown numbers, centered.
_COUNTDOWN_CSS = """
<style>
div[data-testid="stMetric"] { text-align: center !important; }
div[data-testid="stMetricValue"] > div {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #0095FF !important;
}
div[data-testid="stMetricLabel"] > div {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    color: #6b7280 !important;
    text-transform: uppercase;
}
div[data-testid="stMetricDelta"] { display: none !important; }
</style>
"""


def _format_countdown(target_datetime: datetime) -> dict[str, int]:
    """Return days/hours/minutes/seconds until target (timezone-aware)."""
    now = datetime.now(target_datetime.tzinfo) if target_datetime.tzinfo else datetime.now()
    delta = target_datetime - now
    if delta.total_seconds() <= 0:
        return {"days": 0, "hours": 0, "minutes": 0, "seconds": 0}
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return {"days": days, "hours": hours, "minutes": minutes, "seconds": seconds}


@st.fragment(run_every=timedelta(seconds=1))
def _render_live_fomc_countdown(next_meeting: datetime) -> None:
    """Refreshes every second — uses st.metric so numbers are always visible regardless of theme."""
    countdown = _format_countdown(next_meeting)
    st.markdown(
        "<p style='color:#6b7280; font-size:0.78rem; font-weight:600; "
        "letter-spacing:0.07em; text-align:right; margin:0 0 0.3rem 0;'>"
        "NEXT FOMC MEETING IN</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days", countdown["days"])
    c2.metric("Hrs", f"{countdown['hours']:02d}")
    c3.metric("Min", f"{countdown['minutes']:02d}")
    c4.metric("Sec", f"{countdown['seconds']:02d}")


def _render_top_section(current_probs: dict[str, float], next_meeting: datetime) -> None:
    """Expected decision (left) + live countdown (right)."""
    decision_labels = {
        "no_change": "No Change",
        "25bps_cut": "25 bps Cut",
        "25bps_hike": "25 bps Hike",
        "50plus_cut": "50+ bps Cut",
        "50plus_hike": "50+ bps Hike",
    }
    max_outcome = max(current_probs, key=current_probs.get)
    max_prob = current_probs[max_outcome]
    decision_text = decision_labels[max_outcome]

    col1, col2 = st.columns([1, 1])

    with col1:
        # Pure self-contained HTML — no Streamlit widgets inside, no black-bar risk.
        st.markdown(
            f"""
            <p style="color:#6b7280; font-size:0.78rem; font-weight:600;
                      letter-spacing:0.07em; margin:0 0 0.4rem 0;">
                EXPECTED DECISION (POLYMARKET)
            </p>
            <div style="display:flex; align-items:center; gap:1.2rem;">
                <span style="font-size:1.8rem; font-weight:700;">{decision_text}</span>
                <div style="position:relative; width:72px; height:72px; flex-shrink:0;">
                    <svg viewBox="0 0 36 36"
                         style="transform:rotate(-90deg); width:72px; height:72px;">
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831
                                           a 15.9155 15.9155 0 0 1 0 -31.831"
                              fill="none" stroke="#e5e7eb" stroke-width="3"/>
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831
                                           a 15.9155 15.9155 0 0 1 0 -31.831"
                              fill="none" stroke="#0095FF" stroke-width="3"
                              stroke-dasharray="{max_prob}, 100"/>
                    </svg>
                    <div style="position:absolute; top:50%; left:50%;
                                transform:translate(-50%,-50%); text-align:center;">
                        <div style="color:#0095FF; font-size:1.1rem;
                                    font-weight:700; line-height:1.15;">
                            {int(max_prob)}%
                        </div>
                        <div style="color:#6b7280; font-size:0.55rem; font-weight:600;">
                            CHANCE
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        _render_live_fomc_countdown(next_meeting)


def _render_probability_bars(current_probs: dict[str, float]) -> None:
    """Horizontal probability bars — theme-safe colors."""
    outcomes = [
        ("no_change",   "No change",   "#0095FF"),
        ("25bps_cut",   "25 bps cut",  "#f97316"),
        ("25bps_hike",  "25 bps hike", "#eab308"),
        ("50plus_cut",  "50+ bps cut", "#f97316"),
        ("50plus_hike", "50+ bps hike","#eab308"),
    ]

    st.markdown(
        """
        <p style="font-size:1.1rem; font-weight:700; margin:0 0 0.4rem 0;">
            Fed decision probabilities (Polymarket)
        </p>
        <p style="color:#6b7280; font-size:0.88rem; margin:0 0 1rem 0; line-height:1.45;">
            Will there be a change in the Fed&rsquo;s target interest rate at the June&nbsp;17, 2026 FOMC meeting?
        </p>
        """,
        unsafe_allow_html=True,
    )

    for outcome_key, label, color in outcomes:
        prob = current_probs.get(outcome_key, 0.0)
        bar_width = max(1, prob)
        st.markdown(
            f"""
            <div style="margin-bottom:0.6rem;">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem;">
                    <span style="font-weight:700; min-width:52px;">{prob:.1f}%</span>
                    <span style="color:#6b7280; font-size:0.88rem;">{label}</span>
                </div>
                <div style="background:#e5e7eb; height:28px; border-radius:6px; overflow:hidden;">
                    <div style="background:{color}; height:100%; width:{bar_width}%;
                                transition:width 0.3s ease;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_historical_chart(df) -> None:
    """Interactive Plotly chart comparing Polymarket vs Our Model."""
    st.markdown(
        """
        <p style="font-size:1.1rem; font-weight:700; margin:1.5rem 0 0.25rem 0;">
            Odds Over Time
        </p>
        <p style="color:#6b7280; font-size:0.88rem; margin:0 0 1rem 0; line-height:1.5;">
            We ran our model all April to see how it compares to Polymarket.
        </p>
        """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    colors = {
        "no_change":   "#0095FF",
        "25bps_cut":   "#f97316",
        "25bps_hike":  "#eab308",
        "50plus_cut":  "#dc2626",
        "50plus_hike": "#dc2626",
    }
    labels = {
        "no_change":   "No change",
        "25bps_cut":   "25 Bps Cut",
        "25bps_hike":  "25 Bps Hike",
        "50plus_cut":  "50+ Bps Cut",
        "50plus_hike": "50+ Bps Hike",
    }

    for outcome in colors:
        df_o = df[df["outcome"] == outcome].sort_values("date")
        fig.add_trace(go.Scatter(
            x=df_o["date"], y=df_o["probability_polymarket"],
            name=f"{labels[outcome]} (Polymarket)",
            line=dict(color=colors[outcome], width=2),
            mode="lines",
            hovertemplate="<b>Polymarket</b>: " + labels[outcome] + "<br>%{y:.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df_o["date"], y=df_o["probability_our_model"],
            name=f"{labels[outcome]} (Our Model)",
            line=dict(color=colors[outcome], width=2, dash="dash"),
            mode="lines",
            hovertemplate="<b>Our Model</b>: " + labels[outcome] + "<br>%{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        font=dict(color="#9ca3af", size=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#2d2d2d", font_size=11),
        xaxis=dict(gridcolor="#2d2d2d", showgrid=True, zeroline=False, tickformat="%b %d"),
        yaxis=dict(gridcolor="#2d2d2d", showgrid=True, zeroline=False, range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0, font=dict(size=10)),
        margin=dict(l=40, r=20, t=20, b=120),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_fed_tracker() -> None:
    """Render the complete Fed tracker header."""
    # Inject CSS once
    st.markdown(_COUNTDOWN_CSS, unsafe_allow_html=True)

    current_probs = fetch_polymarket_current()
    next_meeting = get_next_fomc_meeting()
    historical_df = fetch_polymarket_historical_april()

    st.markdown("---")
    _render_top_section(current_probs, next_meeting)
    st.markdown("---")
    _render_probability_bars(current_probs)
    _render_historical_chart(historical_df)
    st.markdown("---")
