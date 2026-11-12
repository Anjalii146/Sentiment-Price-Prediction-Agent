"""Walk-forward LightGBM model for the HengSeng Sentiment Agent.

Trains a LightGBM Regressor to predict next-day return using walk-forward
validation (expanding window). No random shuffling, no train_test_split —
time order is strictly preserved to guarantee zero look-ahead bias.

Row for Date N uses features from Date N to predict target_next_return
(derived from Close(N+1)). The model never sees future data as a feature.
"""

from __future__ import annotations

from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

# Feature columns used by the model (everything numeric except identifiers/target)
FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_sentiment_score",
    "social_post_count",
    "news_article_count",
    "avg_engagement_weight",
    "rsi_14",
    "vol_10d",
    "ret_5d",
]

TARGET_COLUMN = "target_next_return"


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split model-ready dataframe into X, y, and metadata.

    Metadata keeps ``date`` and ``ticker`` so predictions can be traced back.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all FEATURE_COLUMNS, TARGET_COLUMN, ``date``, ``ticker``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target values.
    meta : pd.DataFrame
        Identifier columns (date, ticker) plus current_close.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    missing += [c for c in [TARGET_COLUMN, "date", "ticker", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    meta = df[["date", "ticker", "close"]].copy()
    meta = meta.rename(columns={"close": "current_close"})

    return X, y, meta


def train_single_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Train a LightGBM Regressor on one train/test split and predict.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train : pd.Series
        Training targets.

    Returns
    -------
    np.ndarray
        Predictions for X_test.
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 5,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
    )

    predictions = model.predict(X_test)
    return np.asarray(predictions)


if __name__ == "__main__":
    from pathlib import Path

    df = pd.read_csv("data/processed/merged_sample.csv")
    X, y, meta = split_features_target(df)

    # Simple chronological split: train on all but last date, test on last date
    test_date = meta["date"].max()
    train_mask = meta["date"] < test_date
    test_mask = ~train_mask

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("[train_model] Not enough data for a single split; skipping.")
    else:
        preds = train_single_split(
            X[train_mask], y[train_mask], X[test_mask]
        )
        out = meta[test_mask].copy()
        out["predicted_next_return"] = preds
        out["predicted_next_close"] = out["current_close"] * (1 + out["predicted_next_return"])
        out["direction_signal"] = np.where(out["predicted_next_return"] > 0, "Bullish", "Bearish")
        print(f"[train_model] Single-split predictions on {test_date}:")
        print(out[["date", "ticker", "current_close", "predicted_next_return", "direction_signal"]])
