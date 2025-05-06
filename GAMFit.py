#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:25:52 2024

@author: zhexingli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import GAM, LinearGAM, PoissonGAM, GammaGAM, s, f

path = '/test/'
file = path + 'per_Summary.txt'
data = pd.read_csv(file,header=None,skiprows=26,delim_whitespace=True,usecols=[7,8],\
                       names=('err','no'))
    
err_avg = np.average(np.array(data['err']).reshape(-1,5),axis=1)
no_avg = np.average(np.array(data['no']).reshape(-1,5),axis=1)

# calculate sigma for the errors
err_sigma = np.std(np.array(data['err']).reshape(-1,5),axis=1)

# uncertainty plot
# with error bars
fig1, ax1 = plt.subplots()
ax1.errorbar(no_avg, err_avg, yerr=err_sigma, fmt="o",mfc='#377eb8',ecolor='grey',\
             mec='none',alpha=0.5,zorder=-100)
#ax1.legend(loc='upper right',fontsize='medium')
ax1.set_xlabel('Number of New Observations')
ax1.set_ylabel('New $P_{max}$ Uncertainty (days)')
    
# without error bars
fig2, ax2 = plt.subplots()
ax2.plot(no_avg,err_avg,'o',mfc='#377eb8',mew=0,alpha=0.5,zorder=-100)
#ax2.legend(loc='upper right',fontsize='medium')
ax2.set_xlabel('Number of New Observations')
ax2.set_ylabel('New $P_{max}$ Uncertainty (days)')

# smooth
# Fit a GAM model
err_reshape = err_avg.reshape(-1,1)
no_reshape = no_avg.reshape(-1,1)
w = 1/err_sigma.reshape(-1,1)

# parameter space for GAM to explore
lams = np.logspace(-2,2,30)
nsplines = np.array([5,10,15,20,25,30])

# GAM model based on Gamma distribution and log link function
gam = GammaGAM(s(0)).gridsearch(no_reshape, err_reshape,\
               weights = w, n_splines=nsplines, lam=lams)
#gam = GammaGAM(s(0)).fit(no_reshape, err_avg_reshape, weights=w)

# uncertainties predicted by GAM
new_un = gam.predict(no_reshape)

# 95% confidence intervals from GAM
ci = gam.confidence_intervals(no_reshape, width=0.95)

# Summary of the model
#print(gam.summary())

ax1.plot(no_reshape, gam.predict(no_reshape), color='#ff7f00')
ax1.fill_between(no_avg, ci[:,0], ci[:,1],color='#ff7f00',linewidth=0,alpha=0.5)
#ax1.legend(loc='upper right',fontsize='medium')
ax2.plot(no_reshape, gam.predict(no_reshape), color='#ff7f00')
ax2.fill_between(no_avg, ci[:,0], ci[:,1],color='#ff7f00',linewidth=0,alpha=0.5)
#ax2.legend(loc='upper right',fontsize='medium')

fig1.savefig(path+'fit1.pdf')
fig2.savefig(path+'fit2.pdf')
