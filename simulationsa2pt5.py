#!/usr/bin/env python
# coding: utf-8

# In[4]:


#set seed, parameters
import numpy as np 
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as sci
np.random.seed(1996)


# In[5]:


S = 1000
N = 2000
beta_0 = 0
beta_1 = 1
mu_x = 0
sigma_x1 = 1
sigma_e = 1


# In[6]:


#run loop and store estimated values of beta
store_beta = np.zeros((S,2))
for i in range(0,S):
    intr = (np.array([np.ones((N,))])).T
    x1 = (np.array([np.random.standard_normal(N,)])).T
    e = (np.array([np.random.standard_normal(N,)])).T
    y = beta_0 + beta_1*x1 + e
    x = np.concatenate((intr,x1),axis=1)
    model = sm.OLS(y,x).fit()
    betahat = model.params
    betahat_sd = model.bse
    store_beta[i,0] = betahat[1]
    store_beta[i,1] = betahat_sd[1]
#first column is beta1 estimate while the second is the standard error of this estimate


# In[7]:


sci.stats.describe(store_beta)


# In[8]:


#plot beta1 estimates across simulations
plt.plot(store_beta[:,0])
#we disregard estimates for beta0 since the true value is zero 


# In[9]:


#plot standard errors for beta1 estimates across simulations
plt.plot(store_beta[:,1])


# In[10]:


#plot a histogram of the beta1 estimates 
plt.hist(store_beta[:,0])
#we notice the distribution takes the for of a bell shape from a normal distribution


# In[19]:


#compare to theory
est_mean = np.mean(store_beta[:,0])
est_var = np.var(store_beta[:,0])
#theoretical variance
true_var = np.dot(sigma_e,np.linalg.inv((np.dot(x.T,x))))
#the variance of beta1 across simulations is identical to the theoretical variance
print(est_var)
print(true_var[1,1])


# In[20]:


#updated sample size


# In[40]:


def simloop(N,S):
    store2_beta = np.zeros((S,2))
    for i in range(0,S):
        intr = (np.array([np.ones((N,))])).T
        x1 = (np.array([np.random.standard_normal(N,)])).T
        e = (np.array([np.random.standard_normal(N,)])).T
        y = beta_0 + beta_1*x1 + e
        x = np.concatenate((intr,x1),axis=1)
        model = sm.OLS(y,x).fit()
        betahat = model.params
        betahat_sd = model.bse
        store2_beta[i,0] = betahat[1]
        store2_beta[i,1] = betahat_sd[1]
    print(sci.stats.describe(store2_beta))
#first column is beta1 estimate while the second is the standard error of this estimate


# In[41]:


simloop(N=100, S=100)


# In[26]:


sci.stats.describe(store1_beta)


# In[43]:


simloop(N=10000, S=10000)


# In[ ]:




