"""HengSeng Quantitative Sentiment Signals — Bloomberg Terminal Dashboard.

Premium financial terminal UI showing price, sentiment, and next-day return
predictions for tracked assets. Reads from CSVs produced by the ingestion and
modeling pipelines.

Zero look-ahead bias: the dashboard only displays data on or before the
current date. Next-day predictions are clearly labeled as forward-looking
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
    page_icon="▌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Bloomberg Terminal CSS -------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg-dark: #0a0a0a;
        --bg-panel: #121212;
        --bg-card: #161616;
        --border: #333333;
        --accent: #F4A30B;
        --accent-dim: #8a6d1b;
        --text: #e6e6e6;
        --text-dim: #999999;
        --bullish: #00d084;
        --bearish: #ff3b30;
        --font-mono: 'Menlo', 'Monaco', 'Consolas', monospace;
    }

    .stApp {
        background-color: var(--bg-dark);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Top ticker tape */
    .ticker-tape {
        background: #000000;
        border-bottom: 2px solid var(--accent);
        padding: 6px 0;
        overflow: hidden;
        white-space: nowrap;
        font-family: var(--font-mono);
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .ticker-tape-inner {
        display: inline-block;
        animation: ticker 30s linear infinite;
    }
    @keyframes ticker {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    .ticker-item {
        display: inline-block;
        margin: 0 24px;
        color: var(--text-dim);
    }
    .ticker-item .sym {
        color: var(--accent);
        margin-right: 6px;
    }
    .ticker-item .price {
        color: var(--bullish);
    }
    .ticker-item .change.pos { color: var(--bullish); }
    .ticker-item .change.neg { color: var(--bearish); }

    .terminal-header {
        background: var(--bg-panel);
        border-bottom: 1px solid var(--border);
        padding: 14px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 18px;
    }
    .terminal-title {
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 1px;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .terminal-title .dot {
        width: 8px;
        height: 8px;
        background: var(--bullish);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--bullish);
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .terminal-status {
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--accent);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .panel {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .panel-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-dim);
        margin-bottom: 12px;
        font-weight: 700;
        border-bottom: 1px solid var(--border);
        padding-bottom: 8px;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-top: 2px solid var(--accent);
        border-radius: 4px;
        padding: 14px 12px;
        text-align: center;
    }
    .metric-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-dim);
        margin-bottom: 6px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: var(--text);
        font-family: var(--font-mono);
        margin-bottom: 4px;
    }
    .metric-sub {
        font-size: 11px;
        color: var(--text-dim);
        font-family: var(--font-mono);
    }
    .signal-bullish { color: var(--bullish); }
    .signal-bearish { color: var(--bearish); }

    .stDataFrame {
        background: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
    }
    .stDataFrame [data-testid="stDataFrame"] {
        background: var(--bg-panel) !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Data loading helpers ---------------------------------------------------
@st.cache_data
def load_price_data() -> pd.DataFrame:
    path = DATA_PROCESSED / "price_sample.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_daily_sentiment() -> pd.DataFrame:
    path = DATA_PROCESSED / "daily_sentiment_sample.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = OUTPUTS / "predictions_sample.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_news() -> pd.DataFrame:
    path = DATA_RAW / "news_sample.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


# --- Terminal header --------------------------------------------------------
def render_header(ticker: str, price_df: pd.DataFrame) -> None:
    ticker_price = (
        price_df[price_df["ticker"] == ticker].sort_values("date")
        if not price_df.empty
        else pd.DataFrame()
    )
    latest_close = (
        float(ticker_price["close"].iloc[-1]) if not ticker_price.empty else 0.0
    )
    prev_close = (
        float(ticker_price["close"].iloc[-2]) if len(ticker_price) > 1 else latest_close
    )
    change = latest_close - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0
    change_class = "pos" if change >= 0 else "neg"
    change_sign = "+" if change >= 0 else ""

    # Ticker tape items for all assets
    tape_items = []
    for tk in sorted(price_df["ticker"].unique()) if not price_df.empty else []:
        tk_df = price_df[price_df["ticker"] == tk].sort_values("date")
        if tk_df.empty:
            continue
        c = float(tk_df["close"].iloc[-1])
        p = float(tk_df["close"].iloc[-2]) if len(tk_df) > 1 else c
        ch = c - p
        ch_pct = (ch / p * 100) if p else 0.0
        cls = "pos" if ch >= 0 else "neg"
        sg = "+" if ch >= 0 else ""
        tape_items.append(
            f'<span class="ticker-item">'
            f'<span class="sym">{tk}</span>'
            f'<span class="price">{c:,.2f}</span>'
            f'<span class="change {cls}">{sg}{ch_pct:+.2f}%</span>'
            f"</span>"
        )
    tape_html = "".join(tape_items)

    st.markdown(
        f"""
        <div class="terminal-header">
            <div class="terminal-title">
                <span class="dot"></span>
                HENGSENG QUANTITATIVE SENTIMENT SIGNALS
            </div>
            <div class="terminal-status">LIVE MARKET FEED</div>
        </div>
        <div class="ticker-tape">
            <div class="ticker-tape-inner">{tape_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Metric cards -----------------------------------------------------------
def render_metric_cards(
    ticker: str,
    price_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> None:
    ticker_price = (
        price_df[price_df["ticker"] == ticker].sort_values("date")
        if not price_df.empty
        else pd.DataFrame()
    )
    latest_close = (
        float(ticker_price["close"].iloc[-1]) if not ticker_price.empty else None
    )
    latest_date = (
        ticker_price["date"].iloc[-1].strftime("%Y-%m-%d")
        if not ticker_price.empty
        else "—"
    )

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
    predicted_return = (
        float(ticker_pred["predicted_next_return"].iloc[-1])
        if not ticker_pred.empty and "predicted_next_return" in ticker_pred.columns
        else 0.0
    )

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
    social_count = (
        int(ticker_sent["social_post_count"].iloc[-1])
        if not ticker_sent.empty and "social_post_count" in ticker_sent.columns
        else 0
    )
    news_count = (
        int(ticker_sent["news_article_count"].iloc[-1])
        if not ticker_sent.empty and "news_article_count" in ticker_sent.columns
        else 0
    )

    signal_class = (
        "signal-bullish" if latest_signal == "Bullish" else "signal-bearish"
    )
    sentiment_color = "#00d084" if latest_sentiment >= 0 else "#ff3b30"

    ret_sign = "+" if predicted_return >= 0 else ""
    ret_color = "signal-bullish" if predicted_return >= 0 else "signal-bearish"

    st.markdown(
        f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">CURRENT PRICE — {ticker}</div>
                <div class="metric-value">{f"{latest_close:,.2f}" if latest_close is not None else "—"}</div>
                <div class="metric-sub">{latest_date} · Close</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">PREDICTED NEXT CLOSE</div>
                <div class="metric-value">{f"{latest_pred_close:,.2f}" if latest_pred_close is not None else "—"}</div>
                <div class="metric-sub">{ret_sign}{predicted_return:+.2%} expected</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">DIRECTIONAL SIGNAL</div>
                <div class="metric-value {signal_class}">{latest_signal}</div>
                <div class="metric-sub">Model confidence signal</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">SENTIMENT SCORE</div>
                <div class="metric-value" style="color:{sentiment_color}">{latest_sentiment:+.2f}</div>
                <div class="metric-sub">{social_count} social · {news_count} news</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Main chart -------------------------------------------------------------
def render_chart(
    ticker: str,
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> None:
    ticker_price = (
        price_df[price_df["ticker"] == ticker].sort_values("date")
        if not price_df.empty
        else pd.DataFrame()
    )
    ticker_sent = (
        sentiment_df[sentiment_df["ticker"] == ticker].sort_values("date")
        if not sentiment_df.empty
        else pd.DataFrame()
    )

    if ticker_price.empty:
        st.info("No price data available for the selected ticker.")
        return

    chart_df = ticker_price.tail(30).copy()

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

    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="OHLC",
            increasing_line_color="#00d084",
            decreasing_line_color="#ff3b30",
            increasing_fillcolor="#00d084",
            decreasing_fillcolor="#ff3b30",
        )
    )

    fig.add_trace(
        go.Bar(
            x=chart_df["date"],
            y=chart_df["engagement_weighted_sentiment"],
            name="Engagement-Weighted Sentiment",
            marker_color="#F4A30B",
            opacity=0.5,
            yaxis="y2",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        font=dict(color="#e6e6e6", family="Menlo, Monaco, Consolas, monospace", size=11),
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=False),
            type="date",
            gridcolor="#222222",
            showgrid=True,
        ),
        yaxis=dict(
            title="Price",
            side="left",
            gridcolor="#222222",
            showgrid=True,
            tickformat=",.0f",
        ),
        yaxis2=dict(
            title="Engagement-Weighted Sentiment",
            side="right",
            overlaying="y",
            gridcolor="#222222",
            showgrid=False,
            tickformat="+.2f",
        ),
        barmode="overlay",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#999999"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=8, r=8, t=8, b=8),
        height=420,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


# --- News table -------------------------------------------------------------
def render_news_table(ticker: str) -> None:
    news_df = load_news()
    if news_df.empty:
        st.info("No news feed data available.")
        return

    ticker_news = (
        news_df[news_df["ticker"] == ticker]
        .sort_values("date", ascending=False)
        .head(10)
    )

    if ticker_news.empty:
        st.info(f"No recent news for {ticker}.")
        return

    if "sentiment_score" not in ticker_news.columns:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()

        def _score(text: str) -> float:
            if not isinstance(text, str) or not text.strip():
                return 0.0
            return float(_analyzer.polarity_scores(text)["compound"])

        ticker_news = ticker_news.copy()
        ticker_news["sentiment_score"] = ticker_news["title"].astype(str).map(_score)

    display_cols = [
        c for c in ["date", "title", "source", "sentiment_score"]
        if c in ticker_news.columns
    ]
    display_df = ticker_news[display_cols].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

    def _sentiment_color(val: float) -> str:
        if val > 0.05:
            return "color: #00d084"
        if val < -0.05:
            return "color: #ff3b30"
        return "color: #999999"

    st.dataframe(
        display_df.style.map(
            _sentiment_color, subset=["sentiment_score"]
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": "Date",
            "title": "Headline",
            "source": "Source",
            "sentiment_score": st.column_config.NumberColumn(
                "Sentiment", format="%+.2f"
            ),
        },
    )


# --- Main layout ------------------------------------------------------------
price_df = load_price_data()
available_tickers: list[str] = []
if not price_df.empty and "ticker" in price_df.columns:
    available_tickers = sorted(price_df["ticker"].unique().tolist())

if not available_tickers:
    st.error("No market data available. Run the ingestion pipeline first.")
    selected_ticker = st.text_input("Ticker", value="0700.HK")
else:
    # Ticker selector (compact, left-aligned)
    col_sel, _ = st.columns([1, 5])
    with col_sel:
        selected_ticker = st.selectbox(
            "Asset",
            options=available_tickers,
            index=0,
            label_visibility="collapsed",
        )

    # MARKET OVERVIEW panel (header + metrics)
    st.markdown(
        """
        <div class="panel">
        <div class="panel-title">MARKET OVERVIEW</div>
        """,
        unsafe_allow_html=True,
    )
    render_header(selected_ticker, price_df)

    predictions_df = load_predictions()
    sentiment_df = load_daily_sentiment()

    render_metric_cards(
        selected_ticker,
        price_df,
        predictions_df=predictions_df,
        sentiment_df=sentiment_df,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # PRICE & SENTIMENT CHART panel
    st.markdown(
        """
        <div class="panel">
        <div class="panel-title">PRICE & SENTIMENT CHART — LAST 30 DAYS</div>
        """,
        unsafe_allow_html=True,
    )
    render_chart(selected_ticker, price_df, sentiment_df)
    st.markdown("</div>", unsafe_allow_html=True)

    # NEWS FEED panel
    st.markdown(
        """
        <div class="panel">
        <div class="panel-title">NEWS FEED & SENTIMENT</div>
        """,
        unsafe_allow_html=True,
    )
    render_news_table(selected_ticker)
    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div style="margin-top:24px; padding:12px; border-top:1px solid #333333;
        font-family:'Menlo','Monaco','Consolas',monospace; font-size:10px;
        color:#666666; text-align:center;">
        HENGSENG SENTIMENT PRICE-PREDICTION AGENT · ZERO LOOK-AHEAD BIAS · INSTITUTIONAL GRADE
        </div>
        """,
        unsafe_allow_html=True,
    )
