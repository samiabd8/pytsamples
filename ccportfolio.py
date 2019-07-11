#!/usr/bin/env python
# coding: utf-8

# In[1]:


#we are interested in creating an optimal portfolio of cryptocurrencies using simulations


# In[2]:


import pandas as pd  
import numpy as np
import datetime
import scipy.optimize as sco
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.DataFrame(pd.read_csv("/Users/samiabd/Documents/datayobit1h.csv"))
df = df.drop(columns='Date')
pd.DataFrame.info(df)
#we use hourly price data over the span of roughly one year


# In[4]:


#initialize parameters
np.random.seed(1996)
avg_r = df.pct_change().mean()
cov = df.pct_change().cov()
pf_num = 10000
rf = 0
hori = 365
#we randomly generate 10000 portfolios and assume the risk-free rate is zero and the time horizon is one year
#both of these can be adjusted as needed


# In[5]:


#portfolio performance function

def pf_perfm(rf,w,avg_r,cov,hori):
    pf_ret = np.sum(avg_r * w) * hori
    pf_sd = np.sqrt(np.dot(w.T, np.dot(cov,w))) * np.sqrt(hori)
    s_ratio = (pf_ret - rf) / pf_sd
    return pf_ret, pf_sd, s_ratio

#this function calculates the key portfolio statistics: return, standard deviation and Sharpe ratio
#the standard deviation is referred to as the portfolio volatility
#the Sharpe ratio is a commonly used measure of performance that adjusts for risk


# In[6]:


#simulate random portfolios

def sim_random_pf(pf_num, avg_r, cov, rf, hori):
    results = np.zeros((len(avg_r)+3, pf_num))
    for i in range(pf_num):
        w = np.random.random(len(avg_r))
        w /= np.sum(w)
        pf_ret, pf_sd, s_ratio = pf_perfm(rf,w,avg_r,cov, hori)
        results[0,i] = pf_ret
        results[1,i] = pf_sd
        results[2,i] = s_ratio
        #iterate through the weight vector and add data to results array
        for j in range(len(w)):
            results[j+3,i] = w[j]
            
    results_df = pd.DataFrame(results.T,columns=['ret','stdev','sharpe','btc', 'bch', 'btg', 'dash', 'doge', 'eso', 'etc', 'eth', 'liz', 'ltc', 'trx', 'waves', 'xem', 'xvg', 'zec']) 
        
    return results_df

#here we simulate the 10000 portfolios with different weights across cryptocurrencies
#the first three columns are the statistics mentioned above
#the other columns are the weights on each cryptocurrency


# In[7]:


results_frame = sim_random_pf(pf_num, avg_r, cov, rf,hori)
results_frame.describe()


# In[8]:


#the results are stored in this dataframe after running the simulations with our selected parameters
#we then plot these simulated portfolios with the calculated Sharpe ratios as the color scale
#the red star is the simulated portfolio with the highest Sharpe ratio
#the yellow star is the one with the lowest standard deviation or portfolio volatility


# In[9]:


max_sharpe = results_frame.iloc[results_frame['sharpe'].idxmax()]
min_sd = results_frame.iloc[results_frame['stdev'].idxmin()]
plt.subplots(figsize=(15,10))
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.colorbar()
plt.scatter(max_sharpe[1],max_sharpe[0],marker=(5,1,0),color='r',s=500)
plt.scatter(min_sd[1],min_sd[0],marker=(5,1,0),color='g',s=500)
plt.show()


# In[10]:


#view the portfolio with the highest Sharpe ratio
max_sharpe.to_frame().T


# In[11]:


#view the portfolio with the lowest standard deviation
min_sd.to_frame().T


# In[12]:


#we would like to optimize a portfolio and must minimize the portfolio with the most negative Sharpe ratio
#since we cannot maximize with the optimizing function, this is effectively the same result

def neg_sr(w, avg_r, cov, rf):
    pf_ret = np.sum(avg_r * w) * hori
    pf_sd = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(hori)
    sharpe_r = (pf_ret - rf) / pf_sd
    return -sharpe_r


# In[13]:


#defining the constraints to be used, can also be adjusted based on preferences
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})


# In[14]:


def max_sratio(avg_r, cov, rf):
    num_cc = len(avg_r)
    args = (avg_r, cov, rf)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_cc))
    result = sco.minimize(neg_sr, num_cc*[1./num_cc,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# In[15]:


optimal_sharpe = max_sratio(avg_r, cov, rf)


# In[16]:


optimal_sharpe


# In[17]:


pd.DataFrame([round(x,2) for x in optimal_sharpe['x']],index=['btc', 'bch', 'btg', 'dash', 'doge', 'eso', 'etc', 'eth', 'liz', 'ltc', 'trx', 'waves', 'xem', 'xvg', 'zec']).T


# In[18]:


#this is the combination of weights used to produce the optimal portfolio

