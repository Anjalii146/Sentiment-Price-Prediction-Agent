"""Generate structured JSON signals from the latest model predictions.

Reads the walk-forward predictions CSV produced by Phase 6 and emits a
machine-readable ``outputs/signals.json`` payload suitable for downstream
consumption (trading system, frontend API, alerts).

Zero look-ahead bias: signals are derived only from predictions made on or
before the latest available date. No future data is used to construct any
signal.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PREDICTIONS_PATH = Path("outputs/predictions_sample.csv")
SIGNALS_OUTPUT = Path("outputs/signals.json")


def load_latest_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Load predictions and return only the latest date's rows.

    Parameters
    ----------
    path : Path
        Path to the predictions CSV.

    Returns
    -------
    pd.DataFrame
        Rows for the most recent date only, sorted by ticker. Empty if the
        file does not exist or contains no data.
    """
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()
    latest = latest.sort_values("ticker").reset_index(drop=True)
    return latest


def build_signal_row(row: pd.Series) -> dict[str, Any]:
    """Convert a single prediction row into a structured signal dict.

    Parameters
    ----------
    row : pd.Series
        One prediction row containing ``ticker``, ``date``, ``current_close``,
        ``predicted_next_return``, ``predicted_next_close``,
        ``direction_signal``.

    Returns
    -------
    dict
        Structured signal for one ticker.
    """
    signal_date = row["date"]
    if hasattr(signal_date, "date"):
        signal_date = signal_date.date()

    return {
        "ticker": str(row["ticker"]),
        "date": str(signal_date),
        "current_close": float(row["current_close"]),
        "predicted_next_return": float(row["predicted_next_return"]),
        "predicted_next_close": float(row["predicted_next_close"]),
        "direction_signal": str(row["direction_signal"]),
    }


def build_signals(predictions_df: pd.DataFrame) -> dict[str, Any]:
    """Build the full structured signals JSON payload.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output from walk_forward_predict (must contain ``date``, ``ticker``,
        ``current_close``, ``predicted_next_return``, ``predicted_next_close``,
        ``direction_signal``).

    Returns
    -------
    dict
        Payload with ``generated_at``, ``latest_date``, ``signals`` list and
        ``metadata``.
    """
    generated_at = datetime.now(timezone.utc).isoformat()

    if predictions_df.empty:
        return {
            "generated_at": generated_at,
            "latest_date": None,
            "signals": [],
            "metadata": {
                "n_signals": 0,
                "tickers": [],
                "note": "No predictions available. Run the modeling pipeline first.",
            },
        }

    latest_date = predictions_df["date"].max()
    if hasattr(latest_date, "date"):
        latest_date_str = str(latest_date.date())
    else:
        latest_date_str = str(latest_date)

    signals = [build_signal_row(row) for _, row in predictions_df.iterrows()]

    return {
        "generated_at": generated_at,
        "latest_date": latest_date_str,
        "signals": signals,
        "metadata": {
            "n_signals": len(signals),
            "tickers": sorted({s["ticker"] for s in signals}),
        },
    }


def save_signals(signals: dict[str, Any], output_path: Path = SIGNALS_OUTPUT) -> Path:
    """Save the signals dict to a pretty-printed JSON file.

    Parameters
    ----------
    signals : dict
        Structured signals payload from build_signals().
    output_path : Path
        Destination JSON path.

    Returns
    -------
    Path
        The path written to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(signals, fh, indent=2, ensure_ascii=False)
    return output_path


if __name__ == "__main__":
    predictions_df = load_latest_predictions()
    signals = build_signals(predictions_df)
    output_path = save_signals(signals)
    print(
        f"[generate_signals] Saved {signals['metadata']['n_signals']} signals "
        f"to {output_path}"
    )
