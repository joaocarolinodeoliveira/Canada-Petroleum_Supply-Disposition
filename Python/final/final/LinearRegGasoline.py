#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools
import warnings
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import numpy as np



# In[2]:


data = pd.read_csv('Oil_Dataset.csv')
data.head()

data['DATE'] = pd.to_datetime(data['DATE'])



plt.plot(data['DATE'], data['GASOLINE'])
plt.xlabel('Date')
plt.ylabel('Gasoline Price')
plt.show()

variables = ['GASOLINE', 'DIESEL']


for variable in variables:
    X = data[['DATE']]
    y = data[variable]
    
    
    model = LinearRegression()
    model_fit = model.fit(X, y)
    
    
    future_dates = pd.DataFrame(pd.date_range(start='2021-05', end='2027-04', freq='M'), columns=['DATE'])
    future_dates['DATE'] = future_dates['DATE'].astype(int)
    
    predictions = model.predict(future_dates[['DATE']])
    
    future_dates['predicted_price'] = predictions
    

    
    # In[137]:
    
    
    future_dates.to_csv(f'LR_future_predictions_{variable}.csv')



