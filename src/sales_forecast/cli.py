"""CLI for data preparation, model training, and forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DEFAULT_TEST_HORIZON, MODEL_CONFIG, SEGMENT_ALIASES
from .data import clean_sales_data, load_sales_excel, weekly_quantity_by_segment
from .model import evaluate_with_holdout, fit_full_and_save, forecast_weeks, load_model_and_metadata


def _canonical_segment(segment: str) -> str:
    key = segment.strip().lower()
    if key in SEGMENT_ALIASES:
        return SEGMENT_ALIASES[key]
    if segment in MODEL_CONFIG:
        return segment
    raise ValueError(f"Unsupported segment: {segment}. Choose one of {list(MODEL_CONFIG)}")


def run_train(args: argparse.Namespace) -> None:
    raw = load_sales_excel(args.data)
    clean = clean_sales_data(raw)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for segment in MODEL_CONFIG:
        series = weekly_quantity_by_segment(clean, segment)
        metrics = evaluate_with_holdout(series, segment, test_horizon=args.test_horizon)
        artifact = fit_full_and_save(series, segment, model_dir, metrics)

        metrics_rows.append(
            {
                "segment": segment,
                "train_size": metrics.train_size,
                "test_size": metrics.test_size,
                "mae": metrics.mae,
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "model_path": str(artifact.model_path),
                "metadata_path": str(artifact.metadata_path),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Training complete. Metrics saved to: {metrics_path}")
    print(metrics_df.to_string(index=False))


def run_predict(args: argparse.Namespace) -> None:
    segment = _canonical_segment(args.segment)
    model, metadata = load_model_and_metadata(args.model_dir, segment)
    forecast_df = forecast_weeks(
        model=model,
        last_observed_week=metadata["last_observed_week"],
        weeks_ahead=args.weeks,
    )

    print(f"Segment: {segment}")
    print(forecast_df.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(output_path, index=False)
        print(f"Forecast saved to: {output_path}")


def run_prepare(args: argparse.Namespace) -> None:
    raw = load_sales_excel(args.data)
    clean = clean_sales_data(raw)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / "cleaned_sales.csv"
    clean.to_csv(cleaned_path, index=False)

    summary_rows = []
    for segment in MODEL_CONFIG:
        s = weekly_quantity_by_segment(clean, segment)
        summary_rows.append(
            {
                "segment": segment,
                "weeks": int(len(s)),
                "min_week": str(s.index.min().date()),
                "max_week": str(s.index.max().date()),
                "avg_weekly_quantity": float(s.mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "weekly_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Cleaned data written to: {cleaned_path}")
    print(f"Weekly summary written to: {summary_path}")
    print(summary_df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sales forecasting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Clean source data and write summaries")
    prepare.add_argument("--data", default="sales_data.xlsx", help="Path to source workbook")
    prepare.add_argument("--output-dir", default="outputs", help="Output directory")
    prepare.set_defaults(func=run_prepare)

    train = subparsers.add_parser("train", help="Train and persist segment models")
    train.add_argument("--data", default="sales_data.xlsx", help="Path to source workbook")
    train.add_argument("--model-dir", default="models", help="Model artifact directory")
    train.add_argument("--output-dir", default="outputs", help="Output directory")
    train.add_argument(
        "--test-horizon",
        type=int,
        default=DEFAULT_TEST_HORIZON,
        help="Number of trailing weeks for holdout evaluation",
    )
    train.set_defaults(func=run_train)

    predict = subparsers.add_parser("predict", help="Forecast future weekly quantity")
    predict.add_argument("--segment", required=True, help="Consumer | Home Office | Corporate")
    predict.add_argument("--weeks", type=int, required=True, help="Forecast horizon in weeks")
    predict.add_argument("--model-dir", default="models", help="Model artifact directory")
    predict.add_argument("--output", default="", help="Optional output CSV path")
    predict.set_defaults(func=run_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
