# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:21:19 2022

"""

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import datetime as dt

from darts.timeseries import TimeSeries as DTS


from darts.models import TCNModel
from darts.utils.likelihood_models import LaplaceLikelihood

from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

from darts.metrics import mape
from darts.metrics import rmse

#%%

def filter_time(df, t_start, t_end):
    """
    Function to filter a dataframe between t_start and t_end where time is the index
    """
    df = df.loc[ (df.index.values>=t_start) & (df.index.values<=t_end)]
    
    return df

def processdata_prices(df, t_start, t_end):
    """
    Function to process prices
    """
    
    # Convert HourUTC to timestamp
    df['HourUTC'] = pd.to_datetime(df['HourUTC'])
    # Remove timezone
    df['HourUTC'] = df['HourUTC'].dt.tz_localize(None)
    # Set HourUTC as index
    df.set_index('HourUTC', inplace = True)
    # Filter the dataframe for time
    df = filter_time(df, t_start, t_end)
    # Keep only the price areas needed
    df = df.loc[(df['PriceArea']=="DK2")]
    # Sort the dataframe
    df = df.sort_index()
    # Remove useless columns
    df = df[['SpotPriceDKK']]
    return df

#%% Set start and end date for data processing
t_start = pd.to_datetime(dt.datetime(2019, 1, 1, 0, 0, 0))
t_end   = pd.to_datetime(dt.datetime(2021, 12, 31, 23, 0, 0))


cwd = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cwd, "data")

# Load electricity price data
price_path = os.path.join(data_dir, "elspotprices_19to21.csv")
df_price = pd.read_csv(price_path)

# Preprocess price data
df_price = processdata_prices(df_price, t_start, t_end)

# Load exogenous variable data (e.g., wind power)
exo_path = os.path.join(data_dir, "exogenousvariables_19to21.csv")
df_exo = pd.read_csv(exo_path)

#%%

# Convert HourUTC to timestamp
df_exo['HourUTC'] = pd.to_datetime(df_exo['HourUTC'])
# Remove timezone
df_exo['HourUTC'] = df_exo['HourUTC'].dt.tz_localize(None)
# Set HourUTC as index
df_exo.set_index('HourUTC', inplace = True)
# Filter the dataframe for time
df_exo = filter_time(df_exo, t_start, t_end)
# Keep only the price areas needed
df_exo = df_exo.loc[(df_exo['PriceArea']=="DK2")]
# Create combined variables
columns = df_exo.columns
df_exo['WindPower'] = np.sum(df_exo[col] for col in columns[2:6])
df_exo['HydroPower'] = df_exo[columns[6]]
df_exo['SolarPower'] = np.sum(df_exo[col] for col in columns[7:])
# Sort the dataframe
df_exo = df_exo.sort_index()
# Remove useless columns
df_exo = df_exo.drop(columns=[col for col in columns])

#%% 
series_price = DTS.from_dataframe(df_price)
series_exo   = DTS.from_dataframe(df_exo)

#%%
#transforming data
scaler = Scaler()

# Transforming the time series objective
series_price_transformed = scaler.fit_transform(series_price)

# Transforming the time series exogenous
series_exo_transformed   = scaler.fit_transform(series_exo)

#%%
# TRAIN - TEST SPLIT
train_test_split = "20211130T23" # data will be split after  
train_val_price_transformed, test_price_transformed = series_price_transformed.split_after(pd.Timestamp(train_test_split))


# TRAIN - VAL SPLIT
train_val_split = "20211115T23" # data will be split after  
train_price_transformed, val_price_transformed = train_val_price_transformed.split_after(pd.Timestamp(train_val_split))

#%%
plt.figure(figsize=(35,10))
train_price_transformed.plot()
val_price_transformed.plot()
test_price_transformed.plot()

#%%
day_series = datetime_attribute_timeseries(series_price_transformed, attribute = 'hour', one_hot = True)
scaler_day = Scaler()
day_series_transformed = scaler_day.fit_transform(day_series)

series_exo_all_transformed = day_series_transformed.stack(series_exo_transformed)

#%% Find best input length value

for days in range(7):
    model_CNN = TCNModel(
        input_chunk_length=days*24,
        output_chunk_length=24,
        n_epochs = 14,
        dropout = 0.1,
        dilation_base = 2,
        weight_norm = True,
        kernel_size = 3,
        num_filters = 8,
        nr_epochs_val_period = 1, 
        random_state = 0,
    )
    
    # Get training and validation loss for each epoch

#%% Model definition
#For more info: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html
model_CNN = TCNModel(
    input_chunk_length=7*24,
    output_chunk_length=24,
    n_epochs = 14,
    dropout = 0.1,
    dilation_base = 2,
    weight_norm = True,
    kernel_size = 3,
    num_filters = 8,
    nr_epochs_val_period = 1, 
    random_state = 0,
)

#%% Model fitting
#model_CNN.fit(series=train_price_transformed, past_covariates=allCov, verbose=True)
model_CNN.fit(series=train_price_transformed,
             past_covariates = series_exo_all_transformed,
             val_series = val_price_transformed,
             val_past_covariates = series_exo_all_transformed,
             verbose=True)


#%% Prediction all at a time
train_test_split = pd.Timestamp('20211130T23')
train_price_transformed, test_price_transformed = series_price_transformed.split_after(train_test_split)

# Second fit
model_CNN.fit(series = train_price_transformed, past_covariates = series_exo_all_transformed, verbose = True)

# past covariants should also cover the last 'input_chunk_length' of data before the test date begins.
window = 24
prediction = model_CNN.predict(window, past_covariates = series_exo_all_transformed ).pd_dataframe()

test_start = pd.Timestamp('20211201')
test_end = test_start + pd.Timedelta(hours = window)
target = series_price_transformed[test_start:test_end].pd_dataframe()
print("RMSE = {:.2f}%".format(rmse(DTS.from_dataframe(target), DTS.from_dataframe(prediction))))

plt.figure(figsize=(8,4))
plt.plot(target, label = "Real price [DKK]")
plt.plot(prediction.SpotPriceDKK, label="Predicted price [DKK] (H=24)")
plt.legend()

#%% Prediction day by day
train_end = pd.Timestamp('20211130T23')
test_start = pd.Timestamp('20211201')
test_end = pd.Timestamp('20211231T23')
prediction = pd.DataFrame()

for day in range(31):
    train_test_split = train_end + pd.Timedelta(days = day)
    train_price_transformed, test_price_transformed = series_price_transformed.split_after(train_test_split)
    
    model_CNN.fit(series = train_price_transformed, past_covariates = series_exo_all_transformed)
    
    day_prediction = model_CNN.predict(24, past_covariates = series_exo_all_transformed ).pd_dataframe()
    prediction = pd.concat([prediction, day_prediction])

    d = day+1
    print("Predicted day %s" %d)

target = series_price_transformed[test_start:test_end].pd_dataframe()

plt.figure(figsize=(8,4))
plt.plot(target, label = "Real price", color = 'black')
plt.plot(prediction, label = "Predicted price", color = 'blue')
plt.legend()

print("RMSE = {:.2f}%".format(rmse(DTS.from_dataframe(target), DTS.from_dataframe(prediction))))
