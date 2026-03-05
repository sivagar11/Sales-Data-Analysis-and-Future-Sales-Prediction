"""Data loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "Order Date",
    "Segment",
    "Category",
    "Product ID",
    "Region",
    "Regional Manager",
    "Sales",
    "Quantity",
    "Profit",
    "Returned",
}


def load_sales_excel(path: str | Path, sheet_name: str = "GS Sales Data") -> pd.DataFrame:
    """Load raw sales data from the project workbook."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw source column names to snake_case."""
    out = df.copy()
    out.columns = [
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        for c in out.columns
    ]
    return out


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in sales data: {sorted(missing)}")


def _coerce_returned(value: object) -> int:
    if pd.isna(value):
        return 0
    s = str(value).strip().lower()
    return 1 if s in {"yes", "1", "true", "y"} else 0


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply notebook-equivalent cleaning in a deterministic way."""
    _ensure_required_columns(df)
    out = df.copy()

    out["Order Date"] = pd.to_datetime(out["Order Date"], errors="coerce")
    out = out.dropna(subset=["Order Date", "Segment"])

    out["Returned"] = out["Returned"].apply(_coerce_returned)

    for col in ["Sales", "Quantity", "Profit"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    # Mirror notebook logic: returned transactions should not contribute to totals.
    returned_mask = out["Returned"] == 1
    out.loc[returned_mask, ["Sales", "Quantity", "Profit"]] = 0.0

    # Impute missing category from Product ID prefix.
    category_map = {
        "OFF": "Office Supplies",
        "FUR": "Furniture",
        "TEC": "Technology",
    }
    missing_category = out["Category"].isna()
    product_prefix = out["Product ID"].astype(str).str[:3]
    inferred_category = product_prefix.map(category_map)
    out.loc[missing_category, "Category"] = inferred_category[missing_category]

    # Impute missing regional manager from region-level mode.
    region_manager_mode = (
        out.dropna(subset=["Regional Manager"])
        .groupby("Region")["Regional Manager"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
    )
    out["Regional Manager"] = out["Regional Manager"].fillna(
        out["Region"].map(region_manager_mode)
    )

    out = normalize_columns(out)
    out["order_week_start"] = out["order_date"].dt.to_period("W").dt.start_time
    out["segment"] = out["segment"].astype(str).str.strip()

    return out


def weekly_quantity_by_segment(df: pd.DataFrame, segment: str) -> pd.Series:
    """Aggregate weekly shipped quantity for one segment."""
    series_df = (
        df.loc[df["segment"] == segment, ["order_week_start", "quantity"]]
        .groupby("order_week_start", as_index=True)["quantity"]
        .sum()
        .sort_index()
    )
    if series_df.empty:
        raise ValueError(f"No rows found for segment: {segment}")
    return series_df.astype(float)
