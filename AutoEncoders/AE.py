#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:27:32 2020

@author: amira
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 

# Preparing the training set and the test set
training_set= pd.read_csv('ml-100k/u1.base' , delimiter='\t' )
training_set = np.array(training_set , dtype = 'int')
test_set= pd.read_csv('ml-100k/u1.test' , delimiter='\t' )
test_set = np.array(test_set , dtype = 'int')

# getting the number of users and movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Convert the data into an array with users in lines and movies in columns
""" because i need a specific sturcture of data """
def convert (data):
    new_data = []
    for id_users in range (1,nb_users + 1):
        id_movies = data[:,1][data[:,0]==id_users]
        id_ratings = data[:,2][data[:,0]==id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into torch tensors
""" convert lists from numpy array to torch tensor """  
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Creating the architecture of the neural network
class SAE(nn.Module): # SAE mean stacked auto encoder
    def __init__(self, ): # this function for build Archeticture 
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20) # """ 1st Encoding fullconnection layer """
        self.fc2 = nn.Linear(20, 10) # """ 2nd  Encoding fullconnection layer """
        self.fc3 = nn.Linear(10, 20) # """ 1st deconding fullconnection layer """
        self.fc4 = nn.Linear(20, nb_movies) # """ 2nd decoding fullconnection layer """
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x)) # """ take nb_moives input vctor and encoded it to 20 output vector"""
        x = self.activation(self.fc2(x)) # """ take 20 input vctor from fc1 and encoded it to 10 output vector"""
        x = self.activation(self.fc3(x)) # """ take 10 input vctor from fc2 and decoded it to 20 output vector"""
        x = self.fc4(x) # """ take 20 input vctor from fc3 and encoded it to nb_moives output vector"""
        return x
sae = SAE() # """ make object from class """
criterion = nn.MSELoss() 
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = input.clone()
    if torch.sum(target.data > 0) > 0:
      output = sae(input)
      target.require_grad = False
      output[target == 0] = 0
      loss = criterion(output, target)
      mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
      loss.backward()
      train_loss += np.sqrt(loss.data*mean_corrector)
      s += 1.
      optimizer.step()
  print('epoch: '+str(epoch)+ ' loss: '+ str(train_loss/s)) 
  
# Testing the SEA
test_loss = 0
s = 0.
for id_user in range(nb_users):
  input = Variable(training_set[id_user]).unsqueeze(0)
  target = Variable(test_set[id_user]).unsqueeze(0)
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.
print('test loss: '+str(test_loss/s))

