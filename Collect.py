#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:27:01 2023

@author: zhexingli
"""

# Retrieve all the data from the big simulation for orbital ephemerides 
# refinement analaysis

import numpy as np
import pandas as pd
import os

path = '/test/'

no = np.arange(1,201,1)
gap = np.arange(0.25,12.25,0.25)
cov = np.arange(0.5,13.5,0.5)

# The following compile and store all the uncertainty data into one single file
# and store it in a traditional data science way: records in the rows, predictor
# variables in the columns

store = pd.DataFrame({'per2': [],
                      'per2_sig': [],
                      'err': [],
                      'err_sig': [],
                      'No': [],
                      'Gap': [],
                      'Cov': []})

for i in range(len(cov)):
    for j in range(len(gap)):
        # file path
        pathf = path + 'Cov' + str(cov[i]).replace('.','-') + '/' + 'Run' + \
               str(j+1) + '/per_Summary.txt'
        if os.path.exists(pathf):
            # ignoring no from 1 to 5 when reading files
            data = pd.read_csv(pathf,header=None,skiprows=26,delim_whitespace=True,\
                               usecols=[4,7,8,9,10],\
                               names=('per2','per2err','No','Gap','Cov'))
            
            err_avg = np.average(np.array(data['per2err']).reshape(-1,5), axis=1)
            err_sigma = np.std(np.array(data['per2err']).reshape(-1,5), axis=1)
            no_avg = np.round(np.average(np.array(data['No']).reshape(-1,5), axis=1),1)
            gap_avg = np.round(np.average(np.array(data['Gap']).reshape(-1,5), axis=1),2)
            cov_avg = np.round(np.average(np.array(data['Cov']).reshape(-1,5), axis=1),1)
            per_avg = np.average(np.array(data['per2']).reshape(-1,5), axis=1)
            per_sigma = np.std(np.array(data['per2']).reshape(-1,5), axis=1)
            
            temp = pd.DataFrame({'per2': per_avg,
                                 'per2_sig': per_sigma,
                                 'err': err_avg,
                                 'err_sig': err_sigma,
                                 'No': no_avg,
                                 'Gap': gap_avg,
                                 'Cov': cov_avg})
            
            store = store.append(temp,ignore_index=True)

store.to_csv(path+'AlldataML.txt',sep='\t',index=True,header=True)



'''
# The following compile and store all the uncertainty data into one matrix (file)
# initialize a matrix to store the values for three variables

store_err = np.zeros((len(cov),len(gap),len(no)))
store_sig = np.zeros((len(cov),len(gap),len(no)))

for i in range(len(cov)):
    path1 = path + 'Cov' + str(cov[i]).replace('.','-') + '/'
    for j in range(len(gap)):
        path2 = path1 + 'Run' + str(j+1) + '/per_Summary.txt'
        if os.path.exists(path2):
            data = pd.read_csv(path2,header=None,skiprows=1,delim_whitespace=True,\
                               usecols=[4,7],names=('per2','per2err'))
            err = data['per2err']   # all the errors from no 1 to 200
            # average every five values that have the same no
            try:
                # calculate average of the errors and sigma of errors
                err_avg = np.average(np.array(err).reshape(-1,5),axis=1)
                err_sigma = np.std(np.array(err).reshape(-1,5),axis=1)
            except:
                print(path2)
            
            if len(err_avg) != 200 or len(err_sigma) != 200:
                print(f'Data shape incorrect for {path2}.')
                
            store_err[i][j] = err_avg
            store_sig[i][j] = err_sigma
        else:
            print(f'Path doesn\'t exist for {path2}.')
            
# Convert the matrix into a 2d array for storing
store2d_err = store_err.reshape(len(cov),-1)
store2d_sig = store_sig.reshape(len(cov),-1)

# save average error and sigma
np.savetxt(path+'Alldata_err.txt',store2d_err)
np.savetxt(path+'Alldata_sigma.txt',store2d_sig)
'''


