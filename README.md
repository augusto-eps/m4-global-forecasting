# Global Time Series Forecasting with M4 Data

This project explores **global forecasting models** trained across multiple time series, with a focus on situations commonly found in industry:

- Short historical windows
- Strong but heterogeneous seasonality
- Many related time series sharing similar dynamics

Instead of fitting one model per series (ARIMA, Prophet, etc.), we investigate whether **shared patterns** across series can improve forecast accuracy.

---

## Why M4?

The M4 dataset is particularly well suited for this problem because:

- Time series are **positive and stable**
- Each series has **limited history**
- Series are naturally grouped by frequency (Yearly, Quarterly, Monthly, Weekly, Daily)
- It is a standard benchmark used in forecasting research

---

## Project Structure

```text
m4-global-forecasting/
├── data/
│   ├── raw/            # Original M4 files
│   ├── processed/      # Cleaned / transformed datasets
│   └── README.md       # Data documentation
├── src/
│   ├── data/           # Data loading scripts
│   ├── features/       # Feature engineering
│   ├── models/         # Forecasting models
│   └── evaluation/     # Metrics and validation
├── notebooks/          # Exploratory analysis
├── requirements.txt
└── README.md
