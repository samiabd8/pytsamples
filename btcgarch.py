#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:17:37 2020

@author: samiabd
"""

import os
os.getcwd()
os.chdir("/Users/samiabd/github")

import numpy as np
import pandas as pd
from pandas import read_csv
import math
import datetime as dt
from arch import arch_model

import warnings
warnings.simplefilter('ignore')

%matplotlib inline
import matplotlib.pyplot as plt

import seaborn
seaborn.set_style('darkgrid')
plt.rc("figure", figsize=(16, 6))
plt.rc("savefig", dpi=90)
plt.rc("font",family="sans-serif")
plt.rc("font",size=14)

np.random.seed(1996)
df = read_csv('~/github/datayobit1h.csv', engine='python')
df.info()
df.describe()
df['Date']
print(df[7050:])
print(df.btc[-5:])


returns = 100 * df['btc'].pct_change()
returns.head()
log_returns = pd.DataFrame(np.log(1 + returns))
log_returns = log_returns.dropna()
log_returns.info()
ax = log_returns.plot()


#AR vs constant mean GARCH models

gm_ar = arch_model(log_returns, p=1, q=1, mean='AR', lags=12, vol='GARCH', dist='skewt')
gm_ar_result = gm_ar.fit(update_freq=5)
print(gm_ar_result.summary())
gm_ar_fig = gm_ar_result.plot();
plt.show()


gm_ar_forecast = gm_ar_result.forecast(horizon = 24)
print(gm_ar_forecast.variance[-1:])
gm_ar_forecast = gm_ar_forecast.variance[-1:]

gm_ar_std_resid = gm_ar_result.resid / gm_ar_result.conditional_volatility
plt.hist(gm_ar_std_resid, range=(-5,5),facecolor = 'orange' ,
         label = 'standardized residuals',log='TRUE')
plt.show()



gm_cm = arch_model(log_returns, p=1, q=1, mean='constant', lags=12, vol='GARCH', dist='skewt')
gm_cm_result = gm_cm.fit(update_freq=5)
print(gm_cm_result.summary())
gm_cm_fig = gm_cm_result.plot();
plt.show()


gm_cm_forecast = gm_cm_result.forecast(horizon = 24)
print(gm_cm_forecast.variance[-1:])
gm_cm_forecast = gm_cm_forecast.variance[-1:]

gm_cm_std_resid = gm_cm_result.resid / gm_cm_result.conditional_volatility
plt.hist(gm_cm_std_resid, range=(-5,5),facecolor = 'green' , 
         label = 'standardized residuals',log='TRUE')
plt.show()


#comparison plot

plt.plot(gm_cm_result.conditional_volatility, color = 'blue', label = 'Constant Mean GARCH Model Volatility')
plt.plot(gm_ar_result.conditional_volatility, color = 'red', label = 'AR Mean GARCH Model Volatility')
plt.legend(loc = 'upper right')
plt.show()

garch_mean_cor = print(np.corrcoef(gm_cm_result.conditional_volatility, gm_cm_result.conditional_volatility)[0,1])







#GARCH vs EGARCH volatility models

egm_ar = arch_model(log_returns, p=1, q=1, mean='AR', lags=12, vol='EGARCH', dist='skewt')
egm_ar_result = egm_ar.fit(update_freq=5)
print(egm_ar_result.summary())
egm_ar_fig = egm_ar_result.plot();
plt.show()



egm_ar_std_resid = egm_ar_result.resid / egm_ar_result.conditional_volatility
plt.hist(egm_ar_std_resid, range=(-5,5),facecolor = 'orange' ,
         label = 'standardized residuals',log='TRUE');
plt.show()



egm_cm = arch_model(log_returns, p=1, q=1, mean='constant', lags=12, vol='EGARCH', dist='skewt')
egm_cm_result = egm_cm.fit(update_freq=5)
print(egm_cm_result.summary())
egm_cm_fig = egm_cm_result.plot();
plt.show()



egm_cm_std_resid = egm_cm_result.resid / egm_cm_result.conditional_volatility
plt.hist(egm_cm_std_resid, range=(-5,5),facecolor = 'green' , 
         label = 'standardized residuals',log='TRUE');
plt.show()



#comparison plot

plt.plot(egm_cm_result.conditional_volatility, color = 'blue', label = 'Constant Mean EGARCH Volatility')
plt.plot(egm_ar_result.conditional_volatility, color = 'red', label = 'AR Mean EGARCH Volatility')
plt.legend(loc = 'upper right')
plt.show()

egarch_mean_cor = print(np.corrcoef(egm_cm_result.conditional_volatility, egm_cm_result.conditional_volatility)[0,1])



#AR GARCH model vs AR EGARCH model
print(gm_ar_result.bic,gm_cm_result.bic)
print(egm_ar_result.bic,egm_cm_result.bic)

print(gm_ar_result.aic,gm_cm_result.aic)
print(egm_ar_result.aic,egm_cm_result.aic)
#AR is better than constant mean regardless of GARCH or EGARCH 
#AR GARCH is the best overall model


#AR GARCH vs AR EGARCH comparison

plt.plot(gm_ar_result.conditional_volatility, color = 'blue', label = 'AR Mean GARCH Volatility')
plt.plot(egm_ar_result.conditional_volatility, color = 'red', label = 'AR Mean EGARCH Volatility')
plt.legend(loc = 'upper right')
plt.show()

#GARCH AR model captures volatility spikes best