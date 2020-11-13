#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:39:13 2020

@author: amira
"""
# Recurrent Neural Network

# Data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
""" 1:2 means select 1st column but we can't use only 1 because we don't want it as vector""" 

# Feature scale
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range =(0,1))
training_set_scaled = sc.fit_transform(training_set)
"""feature_range =(0,1) because value of Normaliztion feature scale is between 0 and 1"""

# Creating a data structure with 60 timesteps and 1 output
""" this means we take 60 google stock and give one o/p to next node"""
x_train = []  # x, y empty list x will contain 60 previous google stock and y will contain 60 next google stock
y_train = []
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train) 
""" np.array this used to convert x,y to array """

# Reshaping 
""" Adding some dimension to structure """
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
"""1st dimension number of google stock ,2nd dimension number of times step, 3rd dimension number of indicators"""


# RNN Building
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

# Initializing The RNN
regressor = Sequential()  # we will predict continuous output

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences =True , input_shape =(x_train.shape[1], 1)))
""" return_sequences = True when we will add anthor LSTM layer"""
regressor.add(Dropout(0.2))
"""Dropout = 0.2 means that 20% of the neruons will be ignored dring the training """

# Adding second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences =True ))
regressor.add(Dropout(0.2))

# Adding third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences =True ))
regressor.add(Dropout(0.2))

# Adding fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding fourth LSTM layer and some Dropout regularization
regressor.add(Dense(units = 1))

# Compiling The RNN
regressor.compile(optimizer ='adam', loss ='mean_squared_error')

# Fitting The RNN to The training set
regressor.fit(x_train, y_train,epochs = 100, batch_size = 32 )

# Making the prediction and visualising the result

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')