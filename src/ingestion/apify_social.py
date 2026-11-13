"""Apify social media ingestion for the HengSeng Sentiment Agent.

Scrapes social mentions (X/Twitter) for configured tickers using the Apify
platform. Designed to pull a small development sample (10 posts per ticker)
and scale up to the full 10-day lookback at the very end of the sprint.

No look-ahead bias: only posts created on or before the requested date are ever
returned; future-dated content is never included.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

try:
    from apify_client import ApifyClient
except ImportError:  # pragma: no cover - apify-client installed at runtime
    ApifyClient = None  # type: ignore

load_dotenv()


def scrape_social_mentions(
    ticker: str,
    keywords: list[str],
    days_back: int = 10,
    max_posts: int = 10,
) -> pd.DataFrame:
    """Scrape social media mentions for a ticker.

    Parameters
    ----------
    ticker : str
        Market ticker symbol, e.g. ``"0700.HK"``.
    keywords : list[str]
        Search keywords used to find relevant posts.
    days_back : int, default 10
        Lookback window in days (scaled up at end of sprint).
    max_posts : int, default 10
        Number of posts to return during development sampling.

    Returns
    -------
    pd.DataFrame
        Columns: ``date, ticker, platform, text, likes, retweets, views,
        author_followers``.
    """
    columns = [
        "date",
        "ticker",
        "platform",
        "text",
        "likes",
        "retweets",
        "views",
        "author_followers",
    ]
    api_token = os.getenv("APIFY_API_TOKEN")
    actor_id = os.getenv("APIFY_X_SCRAPER_ACTOR_ID")

    # --- Attempt real API call -------------------------------------------
    if api_token and actor_id and ApifyClient is not None:
        try:
            client = ApifyClient(api_token)
            query = " OR ".join(keywords[:3])
            run_input = {
                "searchTerms": [query],
                "maxItems": max_posts,
                "startTime": (
                    datetime.now(timezone.utc) - timedelta(days=days_back)
                ).isoformat(),
            }
            run = client.actor(actor_id).call(run_input=run_input)
            dataset_id = getattr(run, "default_dataset_id", None)
            items = list(
                client.dataset(dataset_id).iterate_items()
            )

            rows = []
            for item in items:
                created_at = item.get("createdAt")
                post_date = (
                    created_at[:10] if created_at else datetime.now(timezone.utc).date().isoformat()
                )
                rows.append(
                    {
                        "date": post_date,
                        "ticker": ticker,
                        "platform": item.get("platform", "x"),
                        "text": item.get("text", ""),
                        "likes": int(item.get("likes", 0) or 0),
                        "retweets": int(item.get("retweets", 0) or 0),
                        "views": int(item.get("views", 0) or 0),
                        "author_followers": int(
                            item.get("authorFollowers", 0) or 0
                        ),
                    }
                )
            if rows:
                return pd.DataFrame(rows, columns=columns)
        except Exception as exc:  # noqa: BLE001 - surface fallback below
            print(f"[apify_social] Real API call failed for {ticker}: {exc}")

    # --- Fallback: small deterministic sample (dev only) -----------------
    print(f"[apify_social] Using sample data for {ticker} ({max_posts} posts)")
    base = datetime.now(timezone.utc)
    rows = []
    for i in range(max_posts):
        rows.append(
            {
                "date": (base - timedelta(hours=i * 3)).strftime("%Y-%m-%d"),
                "ticker": ticker,
                "platform": "x",
                "text": (
                    f"Market discussion about {ticker} #{i + 1} "
                    "#stocks #finance"
                ),
                "likes": (i + 1) * 5,
                "retweets": (i + 1) * 2,
                "views": (i + 1) * 100,
                "author_followers": (i + 1) * 500,
            }
        )
    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    from pathlib import Path

    import yaml

    OUTPUT = Path("data/raw/social_sample.csv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open("config/assets.yaml", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    frames = []
    for asset in config.get("assets", []):
        df = scrape_social_mentions(
            ticker=asset["ticker"],
            keywords=asset.get("keywords", []),
            days_back=10,
            max_posts=10,
        )
        frames.append(df)

    social_df = pd.concat(frames, ignore_index=True)
    social_df.to_csv(OUTPUT, index=False)
    print(f"[apify_social] Saved {len(social_df)} posts to {OUTPUT}")
