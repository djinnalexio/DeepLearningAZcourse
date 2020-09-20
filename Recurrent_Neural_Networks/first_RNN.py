#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:40:23 2020

@author: djinnalexio

Recurrent Neural Network
"""

"""~~~Data Preprocessing~~~"""
import sys, os
import numpy as np #only numpy arrays can be the input of keras
import pandas as pd
import matplotlib.pyplot as plt

filename='Google_Stock_Price_Train.csv'
try: 
    dataset_train = pd.read_csv(filename)
    print ("file obtained from filename")
except FileNotFoundError:
    dataset_train = pd.read_csv(os.path.join(sys.path[0],filename))
    print ("file obtained from absolute path")

# Obtaining an array of the stock prices

#'training_set = dataset_train.iloc[:, 1].values' would only give us a 1D array
training_set = dataset_train.iloc[:, 1:2].values #gives us one column

# Feature Scaling using normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)# it is recommended to keep the original set separate

# Creating a data structure with n timesteps and 1 output
#telling the RNN to remember the n previous stock previous prices

n_timesteps=60

X_train = [] #contains the 60 previous stock prices
Y_train = [] #contains the next stock price


for i in range(n_timesteps, training_set_scaled.shape[0]):
# we start at n_timesteps because X needs at least that number of previous values
#in this case: 0 to 'n_timesteps-1'
    X_train.append(training_set_scaled[i-n_timesteps:i, 0])#previous values
    Y_train.append(training_set_scaled[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)


#Reshaping to add more indicators
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#for RNNs, the dimensions are (batch_size, timesteps, number of indicators)
#We now have a 3D array.

"""~~~Building the RNN~~~"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout # to prevent overfiting

#Initialising the RNN
regressor = Sequential() #the RNN
#Tip: A classifier predicts categories. A regressor predicts a continuous value.


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM())




"""~~~Making Predictions and Visualizing Results~~~"""