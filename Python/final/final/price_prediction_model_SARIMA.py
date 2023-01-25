#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:59:06 2023

@author: mickymwiti
"""


import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import itertools
import numpy as np

# Load the dataset
df = pd.read_csv('Oil_Dataset.csv')
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')
df.sort_values(by='DATE', inplace=True)


# Set the date column as the index
df.set_index('DATE', inplace=True)

# Create a new dataframe to store the test results
results_df = pd.DataFrame(columns=['variable','param', 'param_seasonal', 'AIC', 'MSE'])

# Define the list of variables to predict
variables = ['GASOLINE', 'DIESEL']

for variable in variables:
    # Time series decomposition
    decomposition = seasonal_decompose(df[variable], model='multiplicative')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Stationarity
    result = adfuller(df[variable])
    print(f'{variable} ADF Statistic:', result[0])
    print(f'{variable} p-value:', result[1])

    # Identifying the order of differencing
    df[f'{variable}_diff'] = df[variable] - df[variable].shift()
    plt.plot(df[f'{variable}_diff'])
    plt.show()

    # Identifying the order of the ARIMA model
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plot_acf(df[f'{variable}_diff'].dropna())
    plt.show()
    plot_pacf(df[f'{variable}_diff'].dropna())
    plt.show()

    # Identifying the seasonal component
    plt.plot(df.groupby(df.index.month)[variable].mean())
    plt.show()

    # Split the data into train and test sets
    train = df[:int(0.8*(len(df)))]
    test = df[int(0.8*(len(df))):]

    # Define the p, d, and q parameters to take on any value between 0 and 2
    p = d = q = range(0, 2)

    # Generate all different combinations of p, d, and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, d, and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    best_aic = float('inf')
    best_pdq = None
    best_seasonal_pdq = None
    best_model = None
    best_model_fit = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                # Fit the SARIMA model
                mod = SARIMAX(df[variable],
                                order=param,
                                seasonal_order=param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

                results = mod.fit()
                print(f'SARIMA{param}x{param_seasonal} - AIC:{results.aic}')

                if results.aic < best_aic:
                    best_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
                    best_model = mod
                    best_model_fit = results
            except:
                continue
    # Get the last date of the dataset
    last_date = df.index[-1]
    
    forecast = best_model_fit.forecast(steps=72)

    # Create a new dataframe to store the predictions
    predictions_df = pd.DataFrame(columns=['date'])
    predictions_df['Forecast'] = forecast

    # Get the predictions for the next 6 years
    #for i in range(1, 6*12+1):
        # Get the next month date
       # next_date = last_date + pd.DateOffset(months=i)
        # Get the prediction for the next month
        #pred = best_model_fit.get_forecast(steps=1, index=pd.date_range(start=next_date, periods=1))
        # Append the prediction to the predictions dataframe
        #predictions_df = predictions_df.append({'date': next_date, 'prediction': pred.predicted_mean[0]}, ignore_index=True)

    # Save the predictions dataframe to a csv file
    predictions_df.to_csv(f'SARIMA_predictions_{variable}.csv', index=False)
    
# Get the accuracy of the model
accuracy_df = pd.DataFrame(columns=['variable', 'MSE', 'RMSE', 'MAE'])
for variable in variables:
    ...
    # Get the accuracy of the model
    pred = best_model_fit.get_prediction(start=test.index[0], end=test.index[-1])
    mse = mean_squared_error(test[variable], pred.predicted_mean)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred.predicted_mean - test[variable]))
    accuracy_df = accuracy_df.append({'variable': variable, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}, ignore_index=True)

accuracy_df.to_csv('SARIMA_accuracy.csv', index=False)








