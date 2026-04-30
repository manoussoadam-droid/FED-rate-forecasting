"""Polymarket-style Fed decision probability tracker component."""

from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from components.polymarket_data import (
    fetch_polymarket_current,
    fetch_polymarket_historical_april,
    get_next_fomc_meeting,
)


def _format_countdown(target_datetime: datetime) -> dict[str, int]:
    """Calculate time remaining until target datetime."""
    now = datetime.now()
    delta = target_datetime - now
    
    if delta.total_seconds() <= 0:
        return {"days": 0, "hours": 0, "minutes": 0, "seconds": 0}
    
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return {
        "days": days,
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds,
    }


def _render_top_section(current_probs: dict[str, float], next_meeting: datetime) -> None:
    """Render the top section with expected decision and countdown timer."""
    # Determine expected decision (highest probability)
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
    
    # Get countdown
    countdown = _format_countdown(next_meeting)
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(
            f"""
            <div style="color:#9ca3af; font-size:0.85rem; font-weight:600; letter-spacing:0.05em; margin-bottom:0.5rem;">
                EXPECTED DECISION (POLYMARKET)
            </div>
            <div style="display:flex; align-items:center; gap:1.5rem;">
                <div style="color:#ffffff; font-size:2rem; font-weight:700;">
                    {decision_text}
                </div>
                <div style="position:relative; width:80px; height:80px;">
                    <svg viewBox="0 0 36 36" style="transform:rotate(-90deg);">
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                              fill="none" stroke="#2d2d2d" stroke-width="3"/>
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                              fill="none" stroke="#0095FF" stroke-width="3"
                              stroke-dasharray="{max_prob}, 100"/>
                    </svg>
                    <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%);">
                        <div style="color:#0095FF; font-size:1.2rem; font-weight:700; text-align:center;">
                            {int(max_prob)}%
                        </div>
                        <div style="color:#9ca3af; font-size:0.6rem; text-align:center;">
                            CHANCE
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="color:#9ca3af; font-size:0.85rem; font-weight:600; letter-spacing:0.05em; margin-bottom:0.5rem; text-align:right;">
                MEETING IN
            </div>
            <div style="display:flex; gap:0.75rem; justify-content:flex-end;">
                <div style="text-align:center;">
                    <div style="color:#ffffff; font-size:2rem; font-weight:700; line-height:1;">
                        {countdown['days']}
                    </div>
                    <div style="color:#9ca3af; font-size:0.7rem; font-weight:600;">
                        DAYS
                    </div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#ffffff; font-size:2rem; font-weight:700; line-height:1;">
                        {countdown['hours']:02d}
                    </div>
                    <div style="color:#9ca3af; font-size:0.7rem; font-weight:600;">
                        HRS
                    </div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#ffffff; font-size:2rem; font-weight:700; line-height:1;">
                        {countdown['minutes']:02d}
                    </div>
                    <div style="color:#9ca3af; font-size:0.7rem; font-weight:600;">
                        MIN
                    </div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#ffffff; font-size:2rem; font-weight:700; line-height:1;">
                        {countdown['seconds']:02d}
                    </div>
                    <div style="color:#9ca3af; font-size:0.7rem; font-weight:600;">
                        SEC
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_probability_bars(current_probs: dict[str, float]) -> None:
    """Render horizontal probability bars for all decision outcomes."""
    # Order and labels
    outcomes = [
        ("no_change", "No change", "#0095FF"),
        ("25bps_cut", "25 bps cut", "#f97316"),
        ("25bps_hike", "25 bps hike", "#eab308"),
        ("50plus_cut", "50+ bps cut", "#f97316"),
        ("50plus_hike", "50+ bps hike", "#eab308"),
    ]
    
    st.markdown(
        """
        <div style="color:#ffffff; font-size:1.2rem; font-weight:700; margin-bottom:0.75rem;">
            Fed Decision Probabilities (Polymarket)
        </div>
        <div style="color:#9ca3af; font-size:0.85rem; margin-bottom:1rem;">
            Wed Jun 17, 2026 FOMC Meeting
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    for outcome_key, label, color in outcomes:
        prob = current_probs.get(outcome_key, 0.0)
        bar_width = max(1, prob)  # Minimum 1% for visibility
        
        st.markdown(
            f"""
            <div style="margin-bottom:0.5rem;">
                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">
                    <div style="color:#ffffff; font-weight:600; min-width:50px;">
                        {prob:.1f}%
                    </div>
                    <div style="color:#9ca3af; font-size:0.9rem;">
                        {label}
                    </div>
                </div>
                <div style="background:#2d2d2d; height:32px; border-radius:6px; overflow:hidden;">
                    <div style="background:{color}; height:100%; width:{bar_width}%; 
                                transition:width 0.3s ease;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_historical_chart(df) -> None:
    """Render interactive Plotly chart with Polymarket and Our Model predictions."""
    st.markdown(
        """
        <div style="color:#ffffff; font-size:1.2rem; font-weight:700; margin-top:2rem; margin-bottom:0.5rem;">
            Odds Over Time
        </div>
        <div style="color:#9ca3af; font-size:0.9rem; margin-bottom:1rem; line-height:1.5;">
            We ran our model all April to see how it compares to Polymarket.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Create figure
    fig = go.Figure()
    
    # Color mapping for outcomes
    colors = {
        "no_change": "#0095FF",
        "25bps_cut": "#f97316",
        "25bps_hike": "#eab308",
        "50plus_cut": "#dc2626",
        "50plus_hike": "#dc2626",
    }
    
    labels = {
        "no_change": "No change",
        "25bps_cut": "25 Bps Cut",
        "25bps_hike": "25 Bps Hike",
        "50plus_cut": "50+ Bps Cut",
        "50plus_hike": "50+ Bps Hike",
    }
    
    # Add Polymarket lines (solid)
    for outcome in ["no_change", "25bps_cut", "25bps_hike", "50plus_cut", "50plus_hike"]:
        df_outcome = df[df["outcome"] == outcome].copy()
        df_outcome = df_outcome.sort_values("date")
        
        fig.add_trace(go.Scatter(
            x=df_outcome["date"],
            y=df_outcome["probability_polymarket"],
            name=f"{labels[outcome]} (Polymarket)",
            line=dict(color=colors[outcome], width=2),
            mode="lines",
            hovertemplate=(
                "<b>Polymarket</b>: " + labels[outcome] + "<br>"
                "%{y:.1f}%"
                "<extra></extra>"
            ),
        ))
    
    # Add Our Model lines (dashed)
    for outcome in ["no_change", "25bps_cut", "25bps_hike", "50plus_cut", "50plus_hike"]:
        df_outcome = df[df["outcome"] == outcome].copy()
        df_outcome = df_outcome.sort_values("date")
        
        fig.add_trace(go.Scatter(
            x=df_outcome["date"],
            y=df_outcome["probability_our_model"],
            name=f"{labels[outcome]} (Our Model)",
            line=dict(color=colors[outcome], width=2, dash="dash"),
            mode="lines",
            hovertemplate=(
                "<b>Our Model</b>: " + labels[outcome] + "<br>"
                "%{y:.1f}%"
                "<extra></extra>"
            ),
        ))
    
    # Update layout
    fig.update_layout(
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="#9ca3af", size=12),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#2d2d2d",
            font_size=11,
            font_family="system-ui, -apple-system, sans-serif",
        ),
        xaxis=dict(
            title="",
            gridcolor="#2d2d2d",
            showgrid=True,
            zeroline=False,
            tickformat="%b %d",
        ),
        yaxis=dict(
            title="",
            gridcolor="#2d2d2d",
            showgrid=True,
            zeroline=False,
            range=[0, 100],
            ticksuffix="%",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        margin=dict(l=40, r=20, t=20, b=120),
        height=450,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_fed_tracker() -> None:
    """Main function to render the complete Fed tracker component."""
    # Fetch data
    current_probs = fetch_polymarket_current()
    next_meeting = get_next_fomc_meeting()
    historical_df = fetch_polymarket_historical_april()
    
    # Render in dark container
    with st.container():
        st.markdown(
            """
            <div style="background:#1a1a1a; border-radius:18px; padding:1.5rem; margin-bottom:1rem;">
            """,
            unsafe_allow_html=True,
        )
        
        # Top section
        _render_top_section(current_probs, next_meeting)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Bottom section (probabilities + chart)
    with st.container():
        st.markdown(
            """
            <div style="background:#1a1a1a; border-radius:18px; padding:1.5rem; margin-bottom:1rem;">
            """,
            unsafe_allow_html=True,
        )
        
        # Probability bars
        _render_probability_bars(current_probs)
        
        # Historical chart
        _render_historical_chart(historical_df)
        
        # Close container
        st.markdown("</div>", unsafe_allow_html=True)
