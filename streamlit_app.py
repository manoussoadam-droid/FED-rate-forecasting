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

from core.config import DATA_DIR
from core.analysis_pipeline import analyze_text
from core.ingest import (
    add_parsed_dates,
    build_combined_eda_frame,
    load_fomc,
    load_speaker,
)

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
    text = st.text_area("Paste text", height=240, placeholder="FOMC or speech excerpt…")
    if st.button("Run local analysis"):
        if text.strip():
            with st.spinner("Analyzing…"):
                out = analyze_text(text)
            st.json(out)
            if out.get("wordcloud_png_base64"):
                import base64

                st.image(base64.b64decode(out["wordcloud_png_base64"]), use_container_width=True)
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
