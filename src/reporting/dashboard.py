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


# --- Metric cards -----------------------------------------------------------
def render_metric_cards(
    ticker: str,
    price_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> None:
    """Render the top-row metric cards: price, predicted price, signal, sentiment.

    Parameters
    ----------
    ticker : str
        Selected ticker.
    price_df : pd.DataFrame
        Price data with columns including ``ticker``, ``date``, ``close``.
    predictions_df : pd.DataFrame
        Predictions with columns including ``ticker``, ``date``,
        ``predicted_next_close``, ``direction_signal``.
    sentiment_df : pd.DataFrame
        Daily sentiment with columns including ``ticker``, ``date``,
        ``daily_sentiment_score``.
    """
    # Latest price for the ticker
    ticker_price = (
        price_df[price_df["ticker"] == ticker].sort_values("date")
        if not price_df.empty
        else pd.DataFrame()
    )
    latest_close = (
        float(ticker_price["close"].iloc[-1]) if not ticker_price.empty else None
    )
    latest_date = (
        ticker_price["date"].iloc[-1] if not ticker_price.empty else None
    )

    # Latest prediction for the ticker (most recent date)
    ticker_pred = (
        predictions_df[predictions_df["ticker"] == ticker].sort_values("date")
        if not predictions_df.empty
        else pd.DataFrame()
    )
    latest_pred_close = (
        float(ticker_pred["predicted_next_close"].iloc[-1])
        if not ticker_pred.empty and "predicted_next_close" in ticker_pred.columns
        else None
    )
    latest_signal = (
        str(ticker_pred["direction_signal"].iloc[-1])
        if not ticker_pred.empty and "direction_signal" in ticker_pred.columns
        else "—"
    )

    # Latest sentiment score for the ticker
    ticker_sent = (
        sentiment_df[sentiment_df["ticker"] == ticker].sort_values("date")
        if not sentiment_df.empty
        else pd.DataFrame()
    )
    latest_sentiment = (
        float(ticker_sent["daily_sentiment_score"].iloc[-1])
        if not ticker_sent.empty and "daily_sentiment_score" in ticker_sent.columns
        else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Current Price ({ticker})</div>
                <div class="metric-value">
                    {f"{latest_close:,.2f}" if latest_close is not None else "—"}
                </div>
                <div class="metric-label">
                    {latest_date.strftime("%Y-%m-%d") if latest_date is not None else "—"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Price (Next Day)</div>
                <div class="metric-value">
                    {f"{latest_pred_close:,.2f}" if latest_pred_close is not None else "—"}
                </div>
                <div class="metric-label">Model forecast</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        signal_class = (
            "signal-bullish" if latest_signal == "Bullish" else "signal-bearish"
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Signal</div>
                <div class="metric-value {signal_class}">{latest_signal}</div>
                <div class="metric-label">Directional signal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        sentiment_color = "#3fb950" if latest_sentiment >= 0 else "#f85149"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Social Sentiment Score</div>
                <div class="metric-value" style="color:{sentiment_color}">
                    {latest_sentiment:+.2f}
                </div>
                <div class="metric-label">Daily aggregated sentiment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Main layout ------------------------------------------------------------
st.title("📈 HengSeng Quantitative Sentiment Signals")
st.caption(
    "Bloomberg Terminal-style sentiment signals for price prediction. "
    "Zero look-ahead bias: predictions use only on/before-date data."
)

if not available_tickers:
    st.warning(
        "No price data available. Run the ingestion and modeling pipeline first "
        "(see README)."
    )
else:
    render_metric_cards(
        selected_ticker,
        price_df,
        predictions_df=load_predictions(),
        sentiment_df=load_daily_sentiment(),
    )

    st.divider()

    # --- Main chart: candlestick + sentiment bars ---------------------------
    st.subheader(f"Price & Sentiment — {selected_ticker} (Last 30 Days)")

    ticker_price = (
        price_df[price_df["ticker"] == selected_ticker].sort_values("date")
        if not price_df.empty
        else pd.DataFrame()
    )
    ticker_sent = (
        sentiment_df[sentiment_df["ticker"] == selected_ticker].sort_values("date")
        if not sentiment_df.empty
        else pd.DataFrame()
    )

    if ticker_price.empty:
        st.info("No price data available for the selected ticker.")
    else:
        # Last 30 days of price data
        chart_df = ticker_price.tail(30).copy()

        # Merge sentiment for the bar overlay
        if not ticker_sent.empty:
            chart_df = chart_df.merge(
                ticker_sent[["date", "daily_sentiment_score", "avg_engagement_weight"]],
                on="date",
                how="left",
            )
            chart_df["engagement_weighted_sentiment"] = (
                chart_df["daily_sentiment_score"].fillna(0)
                * chart_df["avg_engagement_weight"].fillna(0)
            )
        else:
            chart_df["engagement_weighted_sentiment"] = 0.0

        fig = go.Figure()

        # Candlestick trace (primary Y)
        fig.add_trace(
            go.Candlestick(
                x=chart_df["date"],
                open=chart_df["open"],
                high=chart_df["high"],
                low=chart_df["low"],
                close=chart_df["close"],
                name="OHLC",
                increasing_line_color="#3fb950",
                decreasing_line_color="#f85149",
            )
        )

        # Engagement-weighted sentiment bars (secondary Y)
        fig.add_trace(
            go.Bar(
                x=chart_df["date"],
                y=chart_df["engagement_weighted_sentiment"],
                name="Engagement-Weighted Sentiment",
                marker_color="#58a6ff",
                opacity=0.6,
                yaxis="y2",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"),
            xaxis=dict(
                title="Date",
                rangeslider=dict(visible=False),
                type="date",
            ),
            yaxis=dict(
                title="Price",
                side="left",
                gridcolor="#30363d",
            ),
            yaxis2=dict(
                title="Engagement-Weighted Sentiment",
                side="right",
                overlaying="y",
                gridcolor="#30363d",
                showgrid=False,
            ),
            barmode="overlay",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Bottom table: recent news headlines with sentiment ----------------
    st.subheader("Recent News & Sentiment")

    news_df = load_news()
    if news_df.empty:
        st.info("No news data available. Run ingestion first.")
    else:
        ticker_news = (
            news_df[news_df["ticker"] == selected_ticker]
            .sort_values("date", ascending=False)
            .head(10)
        )

        if ticker_news.empty:
            st.info(f"No recent news for {selected_ticker}.")
        else:
            # Score sentiment on title if not already present
            if "sentiment_score" not in ticker_news.columns:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

                _analyzer = SentimentIntensityAnalyzer()

                def _score(text: str) -> float:
                    if not isinstance(text, str) or not text.strip():
                        return 0.0
                    return float(_analyzer.polarity_scores(text)["compound"])

                ticker_news = ticker_news.copy()
                ticker_news["sentiment_score"] = ticker_news["title"].astype(str).map(
                    _score
                )

            display_cols = [
                c for c in ["date", "title", "summary", "source", "sentiment_score"]
                if c in ticker_news.columns
            ]
            display_df = ticker_news[display_cols].copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

            def _sentiment_color(val: float) -> str:
                if val > 0.05:
                    return "color: #3fb950"
                if val < -0.05:
                    return "color: #f85149"
                return "color: #8b949e"

            st.dataframe(
                display_df.style.applymap(
                    _sentiment_color, subset=["sentiment_score"]
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": "Date",
                    "title": "Headline",
                    "summary": "Summary",
                    "source": "Source",
                    "sentiment_score": st.column_config.NumberColumn(
                        "Sentiment", format="%+.2f"
                    ),
                },
            )

    st.divider()
    st.caption(
        "HengSeng Sentiment Price-Prediction Agent | "
        "Zero look-ahead bias | Sample dev data — scale to 10-day lookback"
    )
