# montel-electricity-price-forecasting
# 📊 Time Series Forecasting of Renewable Energy Production

This project is part of **Renewable Energy Forecasting with ML and SARIMAX** in the course *Introduction to Energy System Analytics* (MSc Sustainable Energy, DTU). It focuses on forecasting wind and solar power generation using both classical and deep learning models.

## 📁 Contents

- `Neural network prediction.py` – Implements a Temporal Convolutional Network (TCN) using the `darts` library to predict renewable energy production.
- `Battery predictions.py` – Predicts battery behavior based on forecasted energy generation.
- `SARIMAX prediction.py` – Uses a classical statistical approach (SARIMAX) to predict time series data.

> Note: This assignment was done in group format; this repository was prepared individually for educational and portfolio purposes.

---

## 🔍 Problem Statement

Forecasting renewable energy production is critical for grid planning and energy market operations. This project aims to compare traditional statistical forecasting (SARIMAX) with modern deep learning models (TCN) and assess their performance on wind and solar power time series data.

---

## 📊 Data

The input data consists of hourly measurements of wind and solar power production over a certain time period.

### Expected structure (CSV):
- Columns: `Time`, `Wind Power`, `Solar Power`, plus possible exogenous variables (temperature, etc.)
- Frequency: Hourly
- Format: `.csv` files

> The data path is hardcoded in the scripts. Make sure to update paths to match your local setup.

---

## 🧠 Models

### 1. Temporal Convolutional Network (TCN)
- Implemented with the `darts` time series library
- Learns temporal dependencies in sequential data
- Suitable for long sequences and parallel processing

### 2. SARIMAX
- Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors
- Well-suited for data with strong seasonality and known external effects

### 3. Battery Forecasting (Optional Extension)
- Uses energy generation predictions to simulate battery charging/discharging
- Simple rule-based or regression-based model (details in `Battery predictions.py`)

---

## ⚙️ How to Run

### Prerequisites

Install Python packages:

```bash
pip install darts pandas matplotlib scikit-learn
