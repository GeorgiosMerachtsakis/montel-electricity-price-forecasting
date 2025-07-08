# montel-electricity-price-forecasting

This repository was developed as part of a university assignment focused on forecasting hourly electricity prices in Denmark using both classical time series models and deep learning approaches. The project compares the performance of ARIMA, SARIMA, SARIMAX, and Neural network models for short- and medium-term price predictions.


## Repository Structure

├── SARIMAX_Model.py                 # Classical forecasting models (ARIMA, SARIMA, SARIMAX)
├── Neural network prediction.py       # Deep learning model (Temporal Convolutional Neural Network)
├── elspotprices_19to21.csv         # Historical spot prices (input data)
├── exogenousvariables_19to21.csv   # Wind power generation data (exogenous input)
├── docs/
│   └── Project_Description.pdf  # Project description and methodology
└── README.md


## Assignment: Forecasting Electricity Prices
The goal of this assignment was to build and compare models that can forecast electricity spot prices in the Danish DK2 bidding zone. Predictions were evaluated for both hourly (short-term) and daily (medium-term) horizons.

## Models Implemented

1. Classical Time Series Models
Implemented in SARIMAX_Model.py

These models are based on statistical techniques used to predict future values based on past patterns:

ARIMA: Uses past prices and trends to forecast future values.

SARIMA: Like ARIMA, but also includes seasonality (e.g., daily or weekly patterns).

SARIMAX: Extends SARIMA by also considering external factors — in this case, wind power generation as an input.

We used the pmdarima library to train and evaluate these models on hourly spot price data from 2019 to 2020.

2. Temporal Convolutional Neural Network (TCNN)
Implemented in Neural network prediction.py

This is a deep learning model that learns patterns over time using a type of neural network designed for sequences:

Built with PyTorch and the Darts forecasting library

Learns from both past prices and exogenous inputs like wind generation

Can be used for both single-variable and multi-variable forecasting


## Data Sources

All input data and assignment description are located in [`data/`](./data) and [`docs/`](./docs):
- elspotprices_19to21.csv: Hourly day-ahead electricity prices for DK price zones
- Assignment_Description.pdf: Problem statement and task overview

## Requirements

Python ≥ 3.8
pandas
numpy
matplotlib
pmdarima
darts
torch (PyTorch)
