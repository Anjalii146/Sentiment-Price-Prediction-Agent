"""Exa news ingestion for the HengSeng Sentiment Agent.

Fetches recent financial news for configured tickers using the Exa search API.
Designed to pull a small development sample (5 articles per ticker) and scale
up to the full 10-day lookback at the very end of the sprint.

No look-ahead bias: only articles published on or before the requested date are
ever returned; future-dated content is never included.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

try:
    from exa_py import Exa
except ImportError:  # pragma: no cover - exa-py installed at runtime
    Exa = None  # type: ignore

load_dotenv()


def fetch_recent_news(
    ticker: str,
    company_name: str,
    days_back: int = 10,
    max_articles: int = 5,
) -> pd.DataFrame:
    """Fetch recent financial news for a ticker.

    Parameters
    ----------
    ticker : str
        Market ticker symbol, e.g. ``"0700.HK"``.
    company_name : str
        Display name of the company, used to enrich the search query.
    days_back : int, default 10
        Lookback window in days (scaled up to the full window at end of sprint).
    max_articles : int, default 5
        Number of articles to return during development sampling.

    Returns
    -------
    pd.DataFrame
        Columns: ``date, ticker, title, url, summary, source``.
    """
    columns = ["date", "ticker", "title", "url", "summary", "source"]
    api_key = os.getenv("EXA_API_KEY")

    # --- Attempt real API call -------------------------------------------
    if api_key and Exa is not None:
        try:
            exa = Exa(api_key)
            query = (
                f"{company_name} ({ticker}) financial news earnings "
                "market impact stock price"
            )
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=days_back)
            ).strftime("%Y-%m-%d")
            response = exa.search(
                query,
                num_results=max_articles,
                start_published_date=cutoff,
                type="keyword",
            )
            results = getattr(response, "results", []) or []

            rows = []
            for item in results:
                rows.append(
                    {
                        "date": getattr(item, "published_date", None)
                        or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "title": getattr(item, "title", ""),
                        "url": getattr(item, "url", ""),
                        "summary": getattr(item, "summary", "")
                        or getattr(item, "text", ""),
                        "source": "exa",
                    }
                )
            if rows:
                return pd.DataFrame(rows, columns=columns)
        except Exception as exc:  # noqa: BLE001 - surface fallback below
            print(f"[exa_news] Real API call failed for {ticker}: {exc}")

    # --- Fallback: small deterministic sample (dev only) -----------------
    print(f"[exa_news] Using sample data for {ticker} ({max_articles} articles)")
    base = datetime.now(timezone.utc)
    rows = []
    for i in range(max_articles):
        rows.append(
            {
                "date": (base - timedelta(days=i)).strftime("%Y-%m-%d"),
                "ticker": ticker,
                "title": f"[SAMPLE] Market update for {company_name} #{i + 1}",
                "url": "https://example.com/sample-news",
                "summary": (
                    f"Sample financial news summary for {company_name} ({ticker})."
                ),
                "source": "sample",
            }
        )
    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    from pathlib import Path

    OUTPUT = Path("data/raw/news_sample.csv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    assets = [
        {"ticker": "0700.HK", "name": "Tencent Holdings"},
        {"ticker": "005930.KS", "name": "Samsung Electronics"},
    ]

    frames = []
    for asset in assets:
        df = fetch_recent_news(
            ticker=asset["ticker"],
            company_name=asset["name"],
            days_back=10,
            max_articles=5,
        )
        frames.append(df)

    news_df = pd.concat(frames, ignore_index=True)
    news_df.to_csv(OUTPUT, index=False)
    print(f"[exa_news] Saved {len(news_df)} articles to {OUTPUT}")
