#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:09:14 2024

@author: zhexingli
"""

# Train a random forest model on the huge simulation dataset
# Purpose: Smooth the original data, make predictions, and evaluate predictor
# relative importance.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from joblib import dump, load

path = '/test/'

data = pd.read_csv(path+'AlldataML.txt', sep='\s+')

# reshuffle the data to mix up the predictor variables
data_shuffle = data.sample(frac=1).reset_index(drop=True)

# split predictor and target variables
y = data_shuffle['err']
sig = data_shuffle['err_sig']
sigma = (sig - np.min(sig))/(np.max(sig) - np.min(sig)) + max(np.min(sig),1)
weight = 1/sigma    # data points weights according to sigma values
X = data_shuffle.iloc[:,4:]

# split training and test data
X_train, X_test, y_train, y_test, w_train, w_test = \
    train_test_split(X, y, weight, test_size=0.25, random_state=42)


# Scale the data so they're easy to work with in the model (no need for random forest)
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(X_train)
scaled_test_data = scaler.transform(X_test)


# fit random forest with a base model first
rf = RandomForestRegressor(random_state=42, oob_score=True)
rf.fit(X_train, y_train, sample_weight = w_train)

# check base model performance with MSE
prediction = rf.predict(X_test)
mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)

print(mse)
print(rmse)

# train the model with grid search and cross validation
rf = RandomForestRegressor(random_state=42)

# grid search parameters
cv_params_rf = {'max_depth':[2,4,6,8,10,20,None],
                'min_samples_leaf':[1,2,5,10,20],
                'min_samples_split':[2,4,6,8,10],
                'max_features':[1,2,3],
                'n_estimators':[100,200,300,400,500],
                'min_impurity_decrease': [0.0, 0.01],
                'oob_score':[True]}

model_rf = GridSearchCV(rf, cv_params_rf, cv=10, scoring='neg_mean_squared_error', \
                        refit='True', n_jobs=-1)

model_rf.fit(X_train, y_train)

print(model_rf.best_params_)
print(model_rf.best_score_)

dump(model_rf, 'RF_model.joblib')


#############################################################
# load the model and test on test data
model_rf = load(path + 'RF_model.joblib')

# train data performance
model_pred_train = model_rf.predict(X_train)
mse_train = mean_squared_error(y_train, model_pred_train)
rmse_train = np.sqrt(mse_train)

print(f'Train Data Mean Squared Error: {mse_train}.')
print(f'Train Data Root Mean Squared Error: {rmse_train}.')

# test data performance
model_pred_test = model_rf.predict(X_test)

mse_test = mean_squared_error(y_test, model_pred_test)
rmse_test = np.sqrt(mse_test)

print(f'Test Data Mean Squared Error: {mse_test}.')
print(f'Test Data Root Mean Squared Error: {rmse_test}.')






