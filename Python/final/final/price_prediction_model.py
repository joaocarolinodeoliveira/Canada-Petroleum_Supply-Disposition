#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:52:20 2023

@author: mickymwiti
"""

import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

def find_best_arima_params(train):
    # Define the p, d, and q parameters to take on any value between 0 and 2
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    best_aic = float('inf')
    best_param = None

    # Generate all different combinations of p, q and q triplets
    for param in pdq:
        try:
            model = ARIMA(train, order=param)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_param = param
        except:
            continue
    return best_param, best_aic

# Load the dataset
df = pd.read_csv('Oil_Dataset.csv')
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')
df.sort_values(by='DATE', inplace=True)

variables = ['Exports', 'Equivalent Products Production']

for variable in variables:

    # split the dataframe 
    df_1 = df[['DATE', variable]]

    
    # Set the date column as the index
    df_1.set_index('DATE', inplace=True)

    # Time series decomposition
    decomposition = seasonal_decompose(df_1[variable], model='multiplicative')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    
    # Stationarity 
    result = adfuller(df_1[variable])
    print(f'{variable} ADF Statistic:', result[0])
    print(f'{variable} p-value:', result[1])
    
  
    # Identifying the order of differencing 
    df_1[f'{variable}_price_diff'] = df_1[variable] - df_1[variable].shift()
    plt.plot(df_1[f'{variable}_price_diff'])
    plt.show()                     
  
    
    # Identifying the order of the ARIMA model
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plot_acf(df_1[f'{variable}_price_diff'].dropna())
    plt.show()
    plot_pacf(df_1[f'{variable}_price_diff'].dropna())
    plt.show()
    
   
    # Identifying the seasonal component 
    plt.plot(df_1.groupby(df_1.index.month)[variable].mean())
    plt.show()       
    
    
    # Split the data into train and test sets
    train = df_1[:int(0.8*(len(df_1)))]
    test = df_1[int(0.8*(len(df_1))):]
    
   
    # Find the best ARIMA parameters
    best_param, best_aic = find_best_arima_params(train[variable])
    print(f'Best Parameters for {variable}: {best_param}')
    print(f'Best AIC for {variable}: {best_aic}')
    
    
    # Fit the ARIMA model
    model = ARIMA(train[variable], order=best_param)
    model_fit = model.fit(disp=0)
    

    # Forecast the value for the next 6 years
    forecast, stderr, conf_int = model_fit.forecast(steps=72)
   
    # Print the forecasted values
    print(forecast)
    
    
    # Print the mean squared error
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
    mse = mean_squared_error(test[variable], predictions)
    mae = mean_absolute_error(test[variable], predictions)
    r2 = r2_score(test[variable], predictions)
    print(f'Test MSE for {variable}: {mse}')
    print(f'Test MAE for {variable}: {mae}')
    print(f'R-Squared for {variable}: {r2}')

    
    # Create a date range for the forecasted values
    one = 1
    start_date = train.index[-1] + timedelta(days=one)
    end_date = start_date + timedelta(days=2190)
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    
 
    # Create a dataframe with the forecasted values
    forecast_df = pd.DataFrame(date_range, columns=['Date'])
    forecast_df['Forecast'] = forecast
    

    # Plot the forecasted values
    plt.plot(forecast_df['Forecast'])
    plt.show()
    

    # Save the evaluation metrics to a csv file
    eval_metrics = {'MSE': mse, 'MAE': mae, 'R-Squared': r2}
    eval_df = pd.DataFrame.from_dict(eval_metrics, orient='index', columns=['Value'])
    eval_df.to_csv(f'ARIMA_evaluation_metrics_{variable}.csv')


    # save the forecasted values to csv file
    forecast_df.to_csv(f'ARIMA_forecasted_prices_{variable}.csv',index=False)
           
                         
                         
                         
                         
                         
                         
                         
                         
                         