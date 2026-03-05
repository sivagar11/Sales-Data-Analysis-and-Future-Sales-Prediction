# Sales Data Analysis and Future Sales Prediction

This project is now packaged as a runnable Python pipeline for cleaning sales data and forecasting weekly quantity by segment using SARIMAX.

## Project Structure

- `sales_data.xlsx`: Source workbook.
- `Analysis.ipynb`: Original exploratory notebook.
- `src/sales_forecast/`: Reusable pipeline code.
- `models/`: Saved trained SARIMAX models (generated).
- `outputs/`: Generated cleaned data, summaries, and metrics.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Alternative:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

## Usage

Clean data and generate quick summaries:

```bash
sales-forecast prepare --data sales_data.xlsx --output-dir outputs
```

Train all segment models and write metrics:

```bash
sales-forecast train --data sales_data.xlsx --model-dir models --output-dir outputs
```

Predict future quantity for one segment:

```bash
sales-forecast predict --segment "Consumer" --weeks 12 --model-dir models --output outputs/consumer_12w.csv
```

## Model Configuration

The CLI uses segment-specific SARIMAX parameters derived from the original notebook:

- `Consumer`: `(2,0,1) x (1,1,0,52)`
- `Home Office`: `(1,0,0) x (0,1,1,52)`
- `Corporate`: `(1,0,1) x (2,1,1,52)`

## Notes

- Returned orders are normalized to zero `Sales`, `Quantity`, and `Profit` before aggregation.
- Weekly forecasting is based on `Order Date` aggregated to week start dates.
- The original Colab notebook remains unchanged for reference.
