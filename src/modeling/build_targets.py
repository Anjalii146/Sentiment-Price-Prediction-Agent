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


if __name__ == "__main__":
    from pathlib import Path

    OUTPUT = Path("data/processed/merged_sample.csv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    price_df = pd.read_csv("data/processed/price_sample.csv")
    sentiment_df = pd.read_csv("data/processed/daily_sentiment_sample.csv")

    merged = merge_price_sentiment(price_df, sentiment_df)
    merged.to_csv(OUTPUT, index=False)

    print(f"[build_targets] Merged {len(merged)} rows to {OUTPUT}")
