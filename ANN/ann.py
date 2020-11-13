#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:50:04 2020

@author: amira
"""

import pandas as pd
import numpy as np
import tensorflow as ts
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Building the ANN

classifier = ts.keras.models.Sequential()
classifier.add(ts.keras.layers.Dense(units= 6, activation= 'relu'))
classifier.add(ts.keras.layers.Dense(units= 6, activation= 'relu'))
classifier.add(ts.keras.layers.Dense(units= 1, activation= 'sigmoid'))

# Training The ANN

classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train , batch_size=32, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
""" To convert probability to true Or false by put threshold """

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)