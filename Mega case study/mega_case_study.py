#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:56:15 2020

@author: amira
"""
# Mega Case Study - Make a Hybrid with the Deeplearning Model

# Identify the Frauds with the SOM 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
import tensorflow as ts

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X =sc.fit_transform(X)

# Training The SOM 
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,7)],mappings[(7,8)]) , axis =0)
frauds = sc.inverse_transform(frauds)

# Going from Unsupervise to Supervised DeepLearning

# Creating the matrix of feature
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range (len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1
        
# Making RNN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Building the ANN

classifier = ts.keras.models.Sequential()
classifier.add(ts.keras.layers.Dense(units= 2, activation= 'relu' , input_dim = 15))
classifier.add(ts.keras.layers.Dense(units= 1, activation= 'sigmoid'))

# Training The ANN

classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(customers, is_fraud , batch_size=1, epochs=2)

# Predicting the probalitities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis =1)
y_pred = y_pred[y_pred[:, 1].argsort()]