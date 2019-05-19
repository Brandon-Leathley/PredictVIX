# -*- coding: utf-8 -*-
"""
Created on Sat May 18 23:17:42 2019

@author: bleat
"""

import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
import math
import statsmodels.api as sm
from pandas.tseries.offsets import BDay
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, pacf

#data sourced from pandas datareader
vix_train=data.DataReader("VIXCLS", "fred", start='2017-01',end='2019-01')['VIXCLS']
vix_test=data.DataReader("VIXCLS", "fred", start='2019-01',end='2019-05-16')['VIXCLS']
#data formatting (necessary for some components of statsmodels)
dates1=pd.date_range(start='2017-01',end=vix_train.index[-1],freq=BDay())
vix_train.index=dates1
dates2=pd.date_range(start='2019-01',end=vix_test.index[-1],freq=BDay())
vix_test.index=dates2
#data cleaning, replace missing values with previous value
def clean(vix):
    vixdata=vix.copy()
    index=list(vix.index)
    for i in range(len(index)):
        if np.isnan(vix.loc[index[i]]):
            if i==0:
                vixdata=vix.drop(index[0])
            else:
                vixdata.loc[index[i]]=vix.loc[index[i-1]]
    return vixdata
            
train=clean(vix_train)
test=clean(vix_test)

#forecast using Holt-Winters exponential smoothing:
pred=test.copy()
fit1=ExponentialSmoothing(np.asarray(list(train)), seasonal_periods=260, trend='add', seasonal='mul', damped=False).fit()
forecast=fit1.forecast(len(test))
count=0
for i in pred.index:
    pred[i]=forecast[count]
    count+=1

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4)) 
plt.plot(train)
plt.plot(test,label='real data')
plt.plot(pred,label='forecast')
plt.legend()
plt.show()
rms=math.sqrt(mean_squared_error(list(test), list(pred)))
print(rms)
#best: 5.58 with 260 periods, additive trend, multiplicative seasons, no damping

#forecast using seasonal ARIMA:
auto_cor=acf(list(train))
plt.plot(auto_cor)
plt.title('Autocorrelation plot',fontsize='x-large')
plt.show()
#high autocorrelation at early lags ==> want high differencing

partial_ac=pacf(list(train))
plt.plot(partial_ac)
plt.title('Partial Autocorrelation plot',fontsize='x-large')
plt.show()
#partial autocorrelation is high at first lag, then cuts off sharply afterwards ==> try 1 AR term

pred=test.copy()
fit1=sm.tsa.statespace.SARIMAX(list(train), order=(1,1,0),seasonal_order=(0,1,0,196)).fit()
forecast=fit1.forecast(len(test))
count=0
for i in pred.index:
    pred[i]=forecast[count]
    count+=1
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.plot(train,label='Input data')
plt.plot(test,label='Real data')
plt.plot(pred,label='Forecast')
plt.legend(fontsize='medium')
plt.title('VIX forecast with SARIMAX',fontsize='x-large')
plt.show()
rms=math.sqrt(mean_squared_error(list(test), list(pred)))
print(rms)
#varying seasonal periods, forecast becomes most accurate around 196 (approx 9 business months)
#with rms value of 2.40