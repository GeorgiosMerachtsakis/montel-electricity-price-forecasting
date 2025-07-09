#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 15:37:38 2022

"""
#%% 1.0 Process data
import os 
import pandas as pd
import pmdarima as pm
from pmdarima import model_selection
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from pmdarima.utils import plot_acf
from pmdarima.utils import plot_pacf
import numpy as np

# DATA IMPORT
cwd = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cwd, "data")
df = pd.read_csv(os.path.join(data_dir, "elspotprices_19to21.csv"))
df_X = pd.read_csv(os.path.join(data_dir, "exogenousvariables_19to21.csv"))
# Convert HourUTC to timestamp
df['HourUTC'] = pd.to_datetime(df['HourUTC'])
df_X['HourUTC'] = pd.to_datetime(df_X['HourUTC'])
# Remove timezone
df['HourUTC'] = df['HourUTC'].dt.tz_localize(None)
df_X['HourUTC'] = df_X['HourUTC'].dt.tz_localize(None)
# Set HourUTC as index
df.set_index('HourUTC', inplace = True)
df_X.set_index('HourUTC', inplace = True)
# Keep only DK2
df = df.loc[df["PriceArea"]=="DK2"]
df_X = df_X.loc[df_X["PriceArea"]=="DK2"]
# Sort the dataframe
df = df.sort_index()
df_X = df_X.sort_index()
# Choose timeframe
df = df.loc[ (df['HourDK']>'2019-01-01T01:00:00') & (df['HourDK']<='2020-01-01T01:00:00')]
df_X = df_X.loc[ (df_X['HourDK']>'2019-01-01T01:00:00') & (df_X['HourDK']<='2020-01-01T01:00:00')]

# Split data into train and test pieces
df_spot = df[['SpotPriceEUR']].to_numpy()
df_wind = df_X[['OffshoreWindGe100MW_MWh']].to_numpy()
train, test = model_selection.train_test_split(df_spot, train_size=8184)
train_X, test_X = model_selection.train_test_split(df_wind, train_size=8184)

# Plot data 
plt.figure(0)
plt.plot(train)
plt.title('Training set data')
plt.xlabel('time (Hours)')
plt.ylabel('Price (Euro)')
plt.legend(('Spot price for DK2', 'Data length', 'Total message length'),
           loc='upper center', shadow=True)
plt.show()

# Plot ACF
plt.figure(1)
plot_acf(train, lags=48, title = "ACF")
plt.show()
# Plot PACF
plt.figure(2)
plot_pacf(train, lags=48, title = "PACF")
plt.show()


#%%####################################   1.1 - ARIMA model   ###############################################
# Model SARIMAX(5, 1, 5)
#arima = pm.auto_arima(train, error_action='ignore', trace=True, 
#                      suppress_warnings=True, maxiter=20,
#                      seasonal=False)
arima = pm.auto_arima(train, start_p= 5, start_q = 5, d = 1,
                      seasonal=False, trace = True)

arima.plot_diagnostics(lags = 24)
print(arima.summary())

#%% Hour Ahead prediction ARIMA
preds_all_arima_hour = []
for t in test:
    # Make prediction
    preds, conf_int = arima.predict(n_periods=1, return_conf_int=True, alpha = 0.05)
    # Append to forecast lists
    preds_all_arima_hour.append(preds)
    # Update the model with new measurements
    arima.update(t)

# Calculate RMS        
rmse_arima_hour = sqrt(mean_squared_error(test, preds_all_arima_hour))
print('RMSE ARIMA Hourly: %.3f' % rmse_arima_hour)
# Plot prediction and test data
plt.figure(3)
plt.plot(range(0,len(test)), test)  
plt.plot(range(0,len(test)), preds_all_arima_hour)  
plt.title('Hourly prediction with ARIMA model')
plt.xlabel('Time (Hours)')
plt.ylabel('Price (Euro)')
plt.legend(('Spot price real', 'Spot price prediction', 'Total message length'),
           loc='best')
plt.show()
#%% Day ahead prediction ARIMA
preds_all_arima_day = np.empty(0)
k = 0
for t in range(24-1):
    # Make prediction
    preds, conf_int = arima.predict(n_periods=24, return_conf_int=True, alpha = 0.05)
    # Append to forecast lists
    #preds_all_arima_day.append(,preds)cl
    preds_all_arima_day = np.append(preds_all_arima_day, preds)
    # Update the model with new measurements
    arima.update(test[k:k+24])
    k = k+24
# Calculate RMS      
rmse_arima_day = sqrt(mean_squared_error(test, preds_all_arima_day))
print('Test RMSE: %.3f' % rmse_arima_day)
# Plot prediction and test data
plt.figure(3)
plt.plot(range(0,len(test)), test)  
plt.plot(range(0,len(test)), preds_all_arima_day)
plt.title('Daily prediction with ARIMA model')  
plt.xlabel('Time (Hours)')
plt.ylabel('Price (Euro)')
plt.legend(('Spot price real', 'Spot price prediction', 'Total message length'),
           loc='best')
plt.show()


#%%####################################   1.2 - SARIMA model   ###############################################
# Best model SARIMA(1,1,0)(2,0,0)[24]
#arimaS = pm.auto_arima(train, error_action='ignore', trace=True, 
#                      suppress_warnings=True, maxiter=20,
#                      seasonal=True, m = 24)
arimaS = pm.auto_arima(train, start_p=1, start_P=2, start_q = 0, start_Q=0, d = 1, D=0,
                      seasonal=True, m = 24, trace = True)

arimaS.plot_diagnostics(lags = 24)
print(arimaS.summary())
#%% Hour Ahead prediction SARIMA
preds_all_arimaS_hour = []
for t in test:
    # Make prediction
    preds, conf_int = arimaS.predict(n_periods=1, return_conf_int=True, alpha = 0.05)
    # Append to forecast lists
    preds_all_arimaS_hour.append(preds)
    # Update the model with new measurements
    arimaS.update(t)

# Calculate RMS        
rmse_arimaS_hour = sqrt(mean_squared_error(test, preds_all_arimaS_hour))
print('RMSE SARIMA Hourly: %.3f' % rmse_arimaS_hour)
# Plot prediction and test data
plt.figure(3)
plt.plot(range(0,len(test)), test)  
plt.plot(range(0,len(test)), preds_all_arimaS_hour)  
plt.title('Hourly prediction with SARIMA model')
plt.xlabel('Time (Hours)')
plt.ylabel('Price (Euro)')
plt.legend(('Spot price real', 'Spot price prediction', 'Total message length'),
           loc='best')
plt.show()
#%% Day ahead prediction SARIMA
preds_all_arimaS_day = np.empty(0)
k = 0
for t in range(24-1):
    # Make prediction
    preds, conf_int = arimaS.predict(n_periods=24, return_conf_int=True, alpha = 0.05)
    # Append to forecast lists
    #preds_all_arima_day.append(,preds)cl
    preds_all_arimaS_day = np.append(preds_all_arimaS_day, preds)
    # Update the model with new measurements
    arimaS.update(test[k:k+24])
    k = k+24
# Calculate RMS      
rmse_arimaS_day = sqrt(mean_squared_error(test, preds_all_arimaS_day))
print('Test RMSE: %.3f' % rmse_arimaS_day)
# Plot prediction and test data
plt.figure(3)
plt.plot(range(0,len(test)), test)  
plt.plot(range(0,len(test)), preds_all_arimaS_day)
plt.title('Daily prediction with SARIMA model')  
plt.xlabel('Time (Hours)')
plt.ylabel('Price (Euro)')
plt.legend(('Spot price real', 'Spot price prediction', 'Total message length'),
           loc='best')
plt.show()

#%%####################################   1.3 - SARIMAX model   ###############################################
arimaSX = pm.auto_arima(train, error_action='ignore', trace=True, 
                      suppress_warnings=True, maxiter=20,
                      seasonal=True, m = 24, exogenous = train_X)

arimaSX.plot_diagnostics(lags = 24)

print(arimaSX.summary())
#%% Hour Ahead prediction SARIMA
preds_all_arimaSX_hour = []
k = 0
for t in test:
    # Make prediction
    preds, conf_int = arimaSX.predict(n_periods=1, X = test_X[k].reshape(-1,1), return_conf_int=True, alpha = 0.05)
    # Append to forecast lists
    preds_all_arimaSX_hour.append(preds)
    # Update the model with new measurements
    arimaSX.update(t,test_X[k].reshape(-1,1))
    print(k)
    k = k+1
# Calculate RMS        
rmse_arimaSX_hour = sqrt(mean_squared_error(test, preds_all_arimaSX_hour))
print('RMSE SARIMA Hourly: %.3f' % rmse_arimaSX_hour)
# Plot prediction and test data
plt.figure(3)
plt.plot(range(0,len(test)), test)  
plt.plot(range(0,len(test)), preds_all_arimaSX_hour)  
plt.title('Hourly prediction with SARIMAX model')
plt.xlabel('Time (Hours)')
plt.ylabel('Price (Euro)')
plt.legend(('Spot price real', 'Spot price prediction', 'Total message length'),
           loc='best')
plt.show()
#%% Day ahead prediction SARIMA
preds_all_arimaSX_day = np.empty(0)
k = 0
for t in range(23):
    # Make prediction
    preds, conf_int = arimaSX.predict(n_periods=24, X = test_X[k:k+24].reshape(-1,1), return_conf_int=True, alpha = 0.05)
    # Append to forecast lists
    #preds_all_arima_day.append(,preds)cl
    preds_all_arimaSX_day = np.append(preds_all_arimaSX_day, preds)
    # Update the model with new measurements
    arimaSX.update(test[k:k+24],test_X[k:k+24].reshape(-1,1))
    k = k+24
#%% Calculate RMS      
rmse_arimaSX_day = sqrt(mean_squared_error(test[0:552], preds_all_arimaSX_day))
print('Test RMSE: %.3f' % rmse_arimaSX_day)
# Plot prediction and test data
plt.figure(3)
plt.plot(range(0,len(test[0:552])), test[0:552])  
plt.plot(range(0,len(test[0:552])), preds_all_arimaSX_day)
plt.title('Daily prediction with SARIMAX model')  
plt.xlabel('Time (Hours)')
plt.ylabel('Price (Euro)')
plt.legend(('Spot price real', 'Spot price prediction', 'Total message length'),
           loc='best')
plt.show()

#%% Save predictions
from numpy import savetxt
savetxt('SARIMAX_prediction_daily.csv', preds_all_arimaSX_day, delimiter=',')
savetxt('SARIMAX_prediction_hourly.csv', preds_all_arimaSX_hour, delimiter=',')
