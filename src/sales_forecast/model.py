"""Model training, evaluation, persistence, and forecasting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from .config import DEFAULT_TEST_HORIZON, MODEL_CONFIG


@dataclass
class EvalResult:
    segment: str
    train_size: int
    test_size: int
    mae: float
    mse: float
    rmse: float


@dataclass
class TrainedModelArtifact:
    segment: str
    model_path: Path
    metadata_path: Path
    metrics: EvalResult


def evaluate_with_holdout(
    series: pd.Series,
    segment: str,
    test_horizon: int = DEFAULT_TEST_HORIZON,
) -> EvalResult:
    """Fit on train split and evaluate on holdout horizon."""
    if segment not in MODEL_CONFIG:
        raise ValueError(f"Unsupported segment: {segment}")
    if len(series) <= test_horizon + 8:
        raise ValueError(
            f"Not enough data for segment={segment}. Need > {test_horizon + 8} points, got {len(series)}"
        )

    params = MODEL_CONFIG[segment]
    train = series.iloc[:-test_horizon]
    test = series.iloc[-test_horizon:]

    model = SARIMAX(
        train,
        order=params["order"],
        seasonal_order=params["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    forecast = model.get_forecast(steps=len(test)).predicted_mean
    mae = float(mean_absolute_error(test, forecast))
    mse = float(mean_squared_error(test, forecast))
    rmse = float(np.sqrt(mse))

    return EvalResult(
        segment=segment,
        train_size=len(train),
        test_size=len(test),
        mae=mae,
        mse=mse,
        rmse=rmse,
    )


def fit_full_and_save(
    series: pd.Series,
    segment: str,
    model_dir: str | Path,
    metrics: EvalResult,
) -> TrainedModelArtifact:
    """Fit final model on full series and persist model + metadata."""
    if segment not in MODEL_CONFIG:
        raise ValueError(f"Unsupported segment: {segment}")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    params = MODEL_CONFIG[segment]
    model = SARIMAX(
        series,
        order=params["order"],
        seasonal_order=params["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    segment_key = segment.lower().replace(" ", "_")
    model_path = model_dir / f"sarimax_{segment_key}.pkl"
    metadata_path = model_dir / f"sarimax_{segment_key}.json"
    model.save(model_path)

    metadata = {
        "segment": segment,
        "order": list(params["order"]),
        "seasonal_order": list(params["seasonal_order"]),
        "last_observed_week": str(series.index.max().date()),
        "n_observations": int(len(series)),
        "metrics": {
            "train_size": metrics.train_size,
            "test_size": metrics.test_size,
            "mae": metrics.mae,
            "mse": metrics.mse,
            "rmse": metrics.rmse,
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainedModelArtifact(
        segment=segment,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics=metrics,
    )


def load_model_and_metadata(model_dir: str | Path, segment: str) -> tuple[SARIMAXResults, dict]:
    """Load persisted model and metadata for one segment."""
    model_dir = Path(model_dir)
    segment_key = segment.lower().replace(" ", "_")
    model_path = model_dir / f"sarimax_{segment_key}.pkl"
    metadata_path = model_dir / f"sarimax_{segment_key}.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts for segment={segment}. Expected {model_path} and {metadata_path}"
        )

    model = SARIMAXResults.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def forecast_weeks(
    model: SARIMAXResults,
    last_observed_week: str,
    weeks_ahead: int,
) -> pd.DataFrame:
    """Forecast weekly quantity and return a dated dataframe."""
    if weeks_ahead <= 0:
        raise ValueError("weeks_ahead must be positive")

    predicted_mean = model.get_forecast(steps=weeks_ahead).predicted_mean
    start = pd.to_datetime(last_observed_week) + pd.Timedelta(days=7)
    weeks = pd.date_range(start=start, periods=weeks_ahead, freq="W-MON")

    return pd.DataFrame(
        {
            "week_start": weeks,
            "predicted_quantity": np.maximum(predicted_mean.values, 0.0),
        }
    )
