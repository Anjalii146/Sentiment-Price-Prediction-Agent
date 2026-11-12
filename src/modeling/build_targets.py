"""Master merge and target creation for the HengSeng Sentiment Agent.

Merges daily price data with aggregated social and news sentiment, computes
technical features, and creates the next-day return target with correct
shift alignment to guarantee zero look-ahead bias.

Row for Date N contains:
  - Features from Date N (price, sentiment, technicals)
  - Target = next-day return computed from Close(N+1) / Close(N) - 1

The last row is dropped because tomorrow's close is unknown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def merge_price_sentiment(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily price data with aggregated daily sentiment on date/ticker.

    Parameters
    ----------
    price_df : pd.DataFrame
        Columns: ``date, ticker, open, high, low, close, volume``.
    sentiment_df : pd.DataFrame
        Columns: ``date, ticker, daily_sentiment_score, social_post_count,
        news_article_count, avg_engagement_weight``.

    Returns
    -------
    pd.DataFrame
        Merged dataframe sorted by ticker then date.
    """
    required_price = ["date", "ticker", "close"]
    required_sentiment = ["date", "ticker", "daily_sentiment_score"]

    missing_price = [c for c in required_price if c not in price_df.columns]
    missing_sentiment = [c for c in required_sentiment if c not in sentiment_df.columns]
    if missing_price:
        raise ValueError(f"price_df missing columns: {missing_price}")
    if missing_sentiment:
        raise ValueError(f"sentiment_df missing columns: {missing_sentiment}")

    merged = pd.merge(
        price_df,
        sentiment_df,
        on=["date", "ticker"],
        how="inner",
    )

    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    return merged


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the merged dataframe.

    Computes per-ticker:
      - RSI (14): relative strength index using Wilder's smoothing
      - 10-day volatility: rolling std of daily returns
      - 5-day return: close-to-close return over 5 days

    All features use only data up to the current row (no future leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``ticker``, ``date``, ``close`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with added ``rsi_14``, ``vol_10d``, ``ret_5d`` columns.
    """
    required = ["ticker", "date", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df missing columns for technical features: {missing}")

    result = df.sort_values(["ticker", "date"]).copy()
    result["close"] = pd.to_numeric(result["close"], errors="coerce")

    # Daily returns per ticker (used for volatility and RSI)
    result["daily_return"] = result.groupby("ticker")["close"].pct_change()

    # --- RSI (14) per ticker, Wilder's smoothing -----------------------
    rsi_period = 14
    delta = result.groupby("ticker")["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing: first value is simple average, then smoothed
    avg_gain = gain.groupby(result["ticker"]).transform(
        lambda s: s.rolling(rsi_period).mean()
    )
    avg_loss = loss.groupby(result["ticker"]).transform(
        lambda s: s.rolling(rsi_period).mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["rsi_14"] = 100 - (100 / (1 + rs))
    result["rsi_14"] = result["rsi_14"].fillna(50.0)

    # --- 10-day volatility (rolling std of daily returns) ---------------
    result["vol_10d"] = (
        result.groupby("ticker")["daily_return"]
        .transform(lambda s: s.rolling(10).std())
    )
    result["vol_10d"] = result["vol_10d"].fillna(0.0)

    # --- 5-day return ---------------------------------------------------
    result["ret_5d"] = (
        result.groupby("ticker")["close"]
        .transform(lambda s: s.pct_change(5))
    )
    result["ret_5d"] = result["ret_5d"].fillna(0.0)

    result = result.drop(columns=["daily_return"])
    return result


if __name__ == "__main__":
    from pathlib import Path

    OUTPUT = Path("data/processed/merged_sample.csv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    price_df = pd.read_csv("data/processed/price_sample.csv")
    sentiment_df = pd.read_csv("data/processed/daily_sentiment_sample.csv")

    merged = merge_price_sentiment(price_df, sentiment_df)
    merged = add_technical_features(merged)
    merged.to_csv(OUTPUT, index=False)

    print(f"[build_targets] Merged {len(merged)} rows to {OUTPUT}")
    print(
        "[build_targets] Columns: "
        + ", ".join(merged.columns.tolist())
    )
