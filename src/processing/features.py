"""Sentiment scoring and feature engineering for the HengSeng Sentiment Agent.

Scores raw text with vaderSentiment, computes engagement weights for social
posts, and aggregates everything into daily per-ticker sentiment features.

No look-ahead bias: all features are computed from data available on or before
the current date; nothing from a future date leaks into today's features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYZER = SentimentIntensityAnalyzer()


def score_sentiment(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Add a ``sentiment_score`` column (-1 to 1) using vaderSentiment.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a text column.
    text_column : str, default "text"
        Name of the column containing the text to score.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an added ``sentiment_score`` column.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")

    def _score(text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        return float(_ANALYZER.polarity_scores(text)["compound"])

    result = df.copy()
    result["sentiment_score"] = result[text_column].astype(str).map(_score)
    return result


def calculate_engagement_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engagement weight for social posts.

    ``engagement_weight = log(1 + likes + retweets) * log(1 + author_followers)``

    Parameters
    ----------
    df : pd.DataFrame
        Social dataframe with columns ``likes``, ``retweets``,
        ``author_followers``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an added ``engagement_weight`` column.
    """
    required = ["likes", "retweets", "author_followers"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for engagement weight: {missing}")

    result = df.copy()
    for col in required:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)

    result["engagement_weight"] = (
        np.log1p(result["likes"] + result["retweets"])
        * np.log1p(result["author_followers"])
    )
    return result


def daily_aggregate_sentiment(
    social_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate social and news sentiment into daily per-ticker features.

    Social posts are aggregated using an engagement-weighted average of
    ``sentiment_score``. News articles are aggregated using a simple average
    of ``sentiment_score``. The two are combined into a single
    ``daily_sentiment_score`` per date/ticker, weighted by the relative
    volume of social vs. news content.

    Parameters
    ----------
    social_df : pd.DataFrame
        Social dataframe (must already have ``sentiment_score`` and
        ``engagement_weight`` columns).
    news_df : pd.DataFrame
        News dataframe (must already have ``sentiment_score`` column).

    Returns
    -------
    pd.DataFrame
        Columns: ``date, ticker, daily_sentiment_score, social_post_count,
        news_article_count, avg_engagement_weight``.
    """
    social_agg = _aggregate_social(social_df)
    news_agg = _aggregate_news(news_df)

    # Outer merge so tickers with only news or only social are still present
    result = pd.merge(
        social_agg,
        news_agg,
        on=["date", "ticker"],
        how="outer",
        suffixes=("_social", "_news"),
    )

    # Fill missing counts with 0; fill missing sentiment components with 0
    for col in ["social_post_count", "news_article_count"]:
        result[col] = result[col].fillna(0).astype(int)
    for col in ["daily_sentiment_social", "daily_sentiment_news"]:
        result[col] = result[col].fillna(0.0)

    # Combined daily sentiment: weight social by post count, news by article count
    social_weight = result["social_post_count"].astype(float)
    news_weight = result["news_article_count"].astype(float)
    total_weight = social_weight + news_weight

    result["daily_sentiment_score"] = 0.0
    mask = total_weight > 0
    result.loc[mask, "daily_sentiment_score"] = (
        result.loc[mask, "daily_sentiment_social"] * social_weight[mask]
        + result.loc[mask, "daily_sentiment_news"] * news_weight[mask]
    ) / total_weight[mask]

    result["avg_engagement_weight"] = result["avg_engagement_weight"].fillna(0.0)

    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    return result[
        [
            "date",
            "ticker",
            "daily_sentiment_score",
            "social_post_count",
            "news_article_count",
            "avg_engagement_weight",
        ]
    ]


def _aggregate_social(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "daily_sentiment_social",
                "social_post_count",
                "avg_engagement_weight",
            ]
        )

    required = ["date", "ticker", "sentiment_score", "engagement_weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Social df missing columns: {missing}")

    df = df.copy()
    df["weighted_sentiment"] = df["sentiment_score"] * df["engagement_weight"]

    grouped = df.groupby(["date", "ticker"]).agg(
        daily_sentiment_social=("weighted_sentiment", "sum"),
        total_weight=("engagement_weight", "sum"),
        social_post_count=("sentiment_score", "size"),
        avg_engagement_weight=("engagement_weight", "mean"),
    ).reset_index()

    grouped["daily_sentiment_social"] = np.where(
        grouped["total_weight"] > 0,
        grouped["daily_sentiment_social"] / grouped["total_weight"],
        0.0,
    )

    return grouped[
        [
            "date",
            "ticker",
            "daily_sentiment_social",
            "social_post_count",
            "avg_engagement_weight",
        ]
    ]


def _aggregate_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "daily_sentiment_news",
                "news_article_count",
            ]
        )

    required = ["date", "ticker", "sentiment_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"News df missing columns: {missing}")

    grouped = (
        df.groupby(["date", "ticker"])
        .agg(
            daily_sentiment_news=("sentiment_score", "mean"),
            news_article_count=("sentiment_score", "size"),
        )
        .reset_index()
    )

    return grouped


if __name__ == "__main__":
    from pathlib import Path

    import yaml

    SOCIAL_INPUT = Path("data/raw/social_sample.csv")
    NEWS_INPUT = Path("data/raw/news_sample.csv")
    OUTPUT = Path("data/processed/daily_sentiment_sample.csv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    social_df = pd.read_csv(SOCIAL_INPUT)
    news_df = pd.read_csv(NEWS_INPUT)

    # Score sentiment
    social_df = score_sentiment(social_df, text_column="text")
    news_df = score_sentiment(news_df, text_column="title")

    # Engagement weight (social only)
    social_df = calculate_engagement_weight(social_df)

    # Daily aggregation
    daily = daily_aggregate_sentiment(social_df, news_df)
    daily.to_csv(OUTPUT, index=False)

    print(f"[features] Scored {len(social_df)} social posts and {len(news_df)} news articles")
    print(f"[features] Saved {len(daily)} daily rows to {OUTPUT}")
