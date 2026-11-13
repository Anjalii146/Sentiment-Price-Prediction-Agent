"""HengSeng Quantitative Sentiment Signals — Streamlit Dashboard.

Bloomberg Terminal-style UI showing price, sentiment, and next-day return
predictions for tracked assets. Reads from the CSVs produced by the ingestion
and modeling pipelines (sample data during dev, full data at end of sprint).

Zero look-ahead bias: the dashboard only displays data on or before the
current date; next-day predictions are clearly labeled as forward-looking
signals and never used as historical features.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# --- Paths ------------------------------------------------------------------
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
OUTPUTS = Path("outputs")

# --- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="HengSeng Quantitative Sentiment Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Dark mode theme --------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .signal-bullish { color: #3fb950; }
    .signal-bearish { color: #f85149; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Data loading helpers ---------------------------------------------------
@st.cache_data
def load_price_data() -> pd.DataFrame:
    """Load processed price data."""
    path = DATA_PROCESSED / "price_sample.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_daily_sentiment() -> pd.DataFrame:
    """Load aggregated daily sentiment."""
    path = DATA_PROCESSED / "daily_sentiment_sample.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Load model predictions."""
    path = OUTPUTS / "predictions_sample.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_news() -> pd.DataFrame:
    """Load raw news with sentiment."""
    path = DATA_RAW / "news_sample.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


# --- Sidebar ----------------------------------------------------------------
st.sidebar.title("📈 HengSeng Signals")
st.sidebar.caption("Quantitative Sentiment Price-Prediction Agent")

available_tickers: list[str] = []
price_df = load_price_data()
if not price_df.empty and "ticker" in price_df.columns:
    available_tickers = sorted(price_df["ticker"].unique().tolist())

if not available_tickers:
    st.sidebar.warning("No price data available. Run ingestion first.")
    selected_ticker = st.sidebar.text_input("Ticker", value="0700.HK")
else:
    selected_ticker = st.sidebar.selectbox(
        "Select Ticker",
        options=available_tickers,
        index=0,
    )

st.sidebar.divider()
st.sidebar.caption("Data: sample dev set (scale to 10-day lookback at end of sprint)")
