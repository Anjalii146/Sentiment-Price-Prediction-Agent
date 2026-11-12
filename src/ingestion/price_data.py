"""Price data ingestion for the HengSeng Sentiment Agent.

Pulls daily OHLCV price data from Yahoo Finance via yfinance for configured
tickers. Returns a tidy DataFrame with one row per date/ticker.

No look-ahead bias: only historical prices on or before the requested date are
ever returned; future prices are never included.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_price_data(ticker: str, days: int = 200) -> pd.DataFrame:
    """Fetch daily OHLCV price data for a ticker.

    Parameters
    ----------
    ticker : str
        Market ticker symbol, e.g. ``"0700.HK"``.
    days : int, default 200
        Number of trading days of history to pull.

    Returns
    -------
    pd.DataFrame
        Columns: ``date, ticker, open, high, low, close, volume``.
        Sorted ascending by date. Empty DataFrame with correct columns if no
        data is returned.
    """
    columns = ["date", "ticker", "open", "high", "low", "close", "volume"]

    try:
        tk = yf.Ticker(ticker)
        history = tk.history(period=f"{days}d", auto_adjust=False)
    except Exception as exc:  # noqa: BLE001 - surface empty df below
        print(f"[price_data] yfinance call failed for {ticker}: {exc}")
        return pd.DataFrame(columns=columns)

    if history is None or history.empty:
        print(f"[price_data] No price data returned for {ticker}")
        return pd.DataFrame(columns=columns)

    df = history.reset_index()
    # Normalize column names to lowercase
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    # Keep only the columns we need, ensure date is a plain date string
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["ticker"] = ticker
    df = df[columns].sort_values("date").reset_index(drop=True)

    # Coerce numerics
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


if __name__ == "__main__":
    from pathlib import Path

    OUTPUT = Path("data/processed/price_sample.csv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    tickers = ["0700.HK", "005930.KS"]
    frames = []
    for ticker in tickers:
        df = fetch_price_data(ticker, days=200)
        frames.append(df)

    price_df = pd.concat(frames, ignore_index=True)
    price_df.to_csv(OUTPUT, index=False)
    print(f"[price_data] Saved {len(price_df)} rows to {OUTPUT}")
