"""Streamlit UI: corpus overview, time series, text analysis. Run: streamlit run streamlit_app.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from core.analysis_pipeline import analyze_text
from core.config import DATA_DIR
from core.ingest import (
    add_parsed_dates,
    build_combined_eda_frame,
    load_fomc,
    load_speaker,
)


def _render_list(title: str, items: list[str]) -> None:
    st.markdown(f"**{title}**")
    for item in items:
        st.markdown(f"- {item}")


def _render_signal_block(title: str, signals: list[dict[str, str]]) -> None:
    st.markdown(f"**{title}**")
    if not signals:
        st.caption("No clear signal phrases detected.")
        return
    for item in signals:
        st.markdown(item["excerpt_html"], unsafe_allow_html=True)


st.set_page_config(page_title="FOMC text tools", layout="wide")
st.title("FOMC text tools")

tab_data, tab_analyze, tab_api = st.tabs(["Corpus & time series", "Analyze text", "Call Flask API"])

with tab_data:
    st.caption(f"DATA_DIR: `{DATA_DIR}`")
    try:
        fomc = load_fomc()
        speaker = load_speaker()
        combined = build_combined_eda_frame(fomc, speaker)
        fomc_d, speaker_d = add_parsed_dates(fomc, speaker)
        st.success(f"Loaded FOMC: {len(fomc)} rows, Speaker: {len(speaker)} rows")
        st.dataframe(combined.head(20), use_container_width=True)

        work = combined.assign(
            date_parsed=pd.to_datetime(
                combined["date"].astype(str).str.replace(r"\.0$", "", regex=True),
                format="%Y%m%d",
                errors="coerce",
            )
        ).dropna(subset=["date_parsed"])
        if work.empty:
            st.warning("No valid YYYYMMDD dates for time series plot.")
        else:
            ts = (
                work.groupby([pd.Grouper(key="date_parsed", freq="ME"), "decision"])
                .size()
                .unstack(fill_value=0)
            )
            st.subheader("Documents per month by decision")
            fig, ax = plt.subplots(figsize=(10, 4))
            ts.plot(ax=ax, marker=".")
            ax.set_xlabel("Month")
            ax.set_ylabel("Count")
            ax.legend(title="decision")
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.error(str(e))

with tab_analyze:
    text = st.text_area("Paste text", height=240, placeholder="FOMC or speech excerpt...")
    if st.button("Run local analysis"):
        if text.strip():
            with st.spinner("Analyzing..."):
                out = analyze_text(text)

            final = out.get("final_classification", {})
            st.subheader("Final Classification")
            st.markdown(f"**{final.get('label', 'Unknown')}** | Confidence: **{final.get('confidence', 'Low')}**")
            if final.get("justification"):
                st.write(final["justification"])

            dashboard = out.get("metrics_dashboard", {})
            col1, col2 = st.columns(2)
            with col1:
                _render_list(
                    "Macroeconomic Signal Overview",
                    dashboard.get("macroeconomic_signal_overview", []),
                )
            with col2:
                _render_list(
                    "Sentiment Indicators",
                    dashboard.get("sentiment_indicators", []),
                )

            st.subheader("Signal Highlights")
            col3, col4 = st.columns(2)
            with col3:
                _render_signal_block(
                    "Hawkish signals",
                    out.get("signal_highlights", {}).get("hawkish", []),
                )
            with col4:
                _render_signal_block(
                    "Dovish signals",
                    out.get("signal_highlights", {}).get("dovish", []),
                )

            summary = out.get("summary_textrank") or out.get("summary_openai")
            if summary:
                st.subheader("Short Summary")
                st.write(summary)

            if out.get("wordcloud_png_base64"):
                import base64

                st.image(base64.b64decode(out["wordcloud_png_base64"]), use_container_width=True)

            with st.expander("Raw analysis payload"):
                st.json(out)
        else:
            st.warning("Enter some text.")

with tab_api:
    base = st.text_input("Flask base URL", value="http://127.0.0.1:5001")
    if st.button("GET /api/v1/sample"):
        try:
            r = requests.get(f"{base.rstrip('/')}/api/v1/sample", timeout=30)
            st.json(r.json() if r.ok else {"status": r.status_code, "body": r.text})
        except Exception as e:
            st.error(str(e))
    st.divider()
    api_text = st.text_area("Text for API", height=120, key="api_text")
    if st.button("POST /api/v1/analyze"):
        try:
            r = requests.post(
                f"{base.rstrip('/')}/api/v1/analyze",
                json={"text": api_text},
                timeout=120,
            )
            st.json(r.json() if r.ok else {"status": r.status_code, "body": r.text})
        except Exception as e:
            st.error(str(e))
