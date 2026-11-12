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


def walk_forward_predict(
    df: pd.DataFrame,
    n_test_days: int = 10,
) -> pd.DataFrame:
    """Walk-forward validation with expanding window.

    For each of the last ``n_test_days`` unique dates in the dataset:
      - Train on all rows with date strictly before the test date (expanding window)
      - Test on rows with date == test date
    This mirrors "train on days 1 to T, test on T+1; move window forward".

    Time order is strictly preserved — no shuffling, no train_test_split —
    guaranteeing zero look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Model-ready dataframe with features, target, date, ticker, close.
    n_test_days : int, default 10
        Number of most recent dates to use as walk-forward test dates.

    Returns
    -------
    pd.DataFrame
        Columns: ``date, ticker, current_close, actual_next_return,
        predicted_next_return, predicted_next_close, direction_signal``.
        Empty DataFrame with correct columns if insufficient data.
    """
    required = ["date", "ticker", "close", TARGET_COLUMN] + FEATURE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    result_columns = [
        "date",
        "ticker",
        "current_close",
        "actual_next_return",
        "predicted_next_return",
        "predicted_next_close",
        "direction_signal",
    ]

    df = df.sort_values(["date", "ticker"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    unique_dates = sorted(df["date"].unique())
    if len(unique_dates) < 2:
        print("[walk_forward] Not enough unique dates for walk-forward validation.")
        return pd.DataFrame(columns=result_columns)

    # Use the last n_test_days unique dates as test dates
    test_dates = unique_dates[-n_test_days:]

    all_predictions = []

    for test_date in test_dates:
        train_mask = df["date"] < test_date
        test_mask = df["date"] == test_date

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        if n_train == 0 or n_test == 0:
            print(
                f"[walk_forward] Skipping {test_date.date()}: "
                f"train={n_train}, test={n_test}"
            )
            continue

        X_train, y_train, _ = split_features_target(df[train_mask])
        X_test, y_test, meta_test = split_features_target(df[test_mask])

        preds = train_single_split(X_train, y_train, X_test)

        fold = meta_test.copy()
        fold["actual_next_return"] = y_test.values
        fold["predicted_next_return"] = preds
        fold["predicted_next_close"] = (
            fold["current_close"] * (1 + fold["predicted_next_return"])
        )
        fold["direction_signal"] = np.where(
            fold["predicted_next_return"] > 0, "Bullish", "Bearish"
        )

        all_predictions.append(fold[result_columns])

    if not all_predictions:
        print("[walk_forward] No predictions generated.")
        return pd.DataFrame(columns=result_columns)

    result = pd.concat(all_predictions, ignore_index=True)
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(
        f"[walk_forward] Generated {len(result)} predictions across "
        f"{result['date'].nunique()} test dates"
    )

    return result


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


def calculate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Fraction of predictions where predicted sign matches actual sign.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Actual and predicted next-day returns.

    Returns
    -------
    float
        Directional accuracy in [0, 1]. Returns 0.0 if inputs are empty.
    """
    if len(y_true) == 0:
        return 0.0
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = np.sign(y_true) == np.sign(y_pred)
    return float(correct.mean())


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error between actual and predicted values.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Actual and predicted next-day returns.

    Returns
    -------
    float
        RMSE. Returns 0.0 if inputs are empty.
    """
    if len(y_true) == 0:
        return 0.0
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute directional accuracy and RMSE from a predictions DataFrame.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must contain ``actual_next_return`` and ``predicted_next_return``.

    Returns
    -------
    dict
        Keys: ``directional_accuracy``, ``rmse``, ``n_predictions``.
    """
    if predictions.empty:
        return {
            "directional_accuracy": 0.0,
            "rmse": 0.0,
            "n_predictions": 0,
        }

    y_true = predictions["actual_next_return"].to_numpy()
    y_pred = predictions["predicted_next_return"].to_numpy()

    return {
        "directional_accuracy": calculate_directional_accuracy(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "n_predictions": int(len(predictions)),
    }


if __name__ == "__main__":
    from pathlib import Path

    OUTPUT = Path("outputs/predictions_sample.csv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/processed/merged_sample.csv")

    predictions = walk_forward_predict(df, n_test_days=10)
    if not predictions.empty:
        metrics = evaluate_predictions(predictions)
        predictions.to_csv(OUTPUT, index=False)
        print(f"[train_model] Saved {len(predictions)} walk-forward predictions to {OUTPUT}")
        print(
            predictions[
                ["date", "ticker", "current_close", "predicted_next_return", "direction_signal"]
            ].to_string(index=False)
        )
        print(f"[train_model] Directional accuracy: {metrics['directional_accuracy']:.3f}")
        print(f"[train_model] RMSE: {metrics['rmse']:.6f}")
        print(f"[train_model] N predictions: {metrics['n_predictions']}")
    else:
        print("[train_model] No walk-forward predictions to save.")
