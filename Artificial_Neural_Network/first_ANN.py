#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 19:54:02 2020

@author: djinnalexio

Artificial Neural Network
"""

#modules used: os, sys, numpy, pandas, tensorflow2, sklearn

"""~~~~Importing the Libraries~~~~"""
import os
import sys
import numpy as np
np.set_printoptions(suppress=True)#prevents from printing in scientific notation
#import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
print ("\nTensorflow version:",tf.__version__,"\n")

"""~~~~Importing the Dataset~~~~"""
fileName = 'Churn_Modelling.csv'
try:    
    dataset = pd.read_csv(os.path.join(sys.path[0],fileName))# use the full path of the file that is in the same directory as the script
    print ("file obtained from absolute path")
except FileNotFoundError:    
    dataset = pd.read_csv(fileName)# if fail, only use the name of the file
    print ("file obtained from filename")


X_Input_data = dataset.iloc[:, 3:-1].values #decide which columns (attributes) will be used in the calculations of the ANN
#the values from column index 3 to the one before last
Y_Real_value = dataset.iloc[:, -1].values #decide the dependent variable(s) we are trying to guess. 
# Here, the results are simply the last column.

print (X_Input_data)
print (Y_Real_value)


"""~~~~Encoding Categorical Data~~~~""" #turning strings into values the computer can calculate with

"""Encoding genders"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_Input_data[:,2] = le.fit_transform(X_Input_data[:,2])
#take all the values in 'Gender' and replace them by an encoded form
#here 'female' and 'male' become 0 and 1
print (X_Input_data)


"""Encoding countries (One Hot Encoding)"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X_Input_data = np.array(ct.fit_transform(X_Input_data))
print (X_Input_data)
#take all the values in the second column 'Geography' and replace them by an encoded form
#turns strings into series of 0s and 1s


"""~~~~Splitting the dataset into the Training set and Test set~~~~"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Input_data, Y_Real_value, test_size = 0.2, random_state = 0)


"""~~~~Feature Scaling~~~~"""#fondamental step for deep learning
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)


"""
PART 2: Building the ANN
"""

"""~~~~Initializing the Artificial Neural Network~~~~"""
ANN = tf.keras.models.Sequential()

"""~~~~Adding input layer and hidden layers~~~~"""
#the input layer will automatically be made of the attributes of the dataset

ANN.add(tf.keras.layers.Dense(units=14, activation='relu'))
# 'units': the numbers of neurons in the layer
# 'activation' : the formula that is being used in the layer
# 'relu' = rectifier function

ANN.add(tf.keras.layers.Dense(units=10, activation='relu'))

ANN.add(tf.keras.layers.Dense(units=6, activation='relu'))

"""~~~~Adding the output layer~~~~"""
ANN.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#since the answer is yes/no, we only need one neuron to get a guess
#if they were 3 or more more possible results, they would be one hot encoded and each result would get a neuron
#'Sigmoid' function gives the result and also the probability

"""
For non-binary classification (more than 2 possibilities)
    activation funstion of the output layer = 'softmax'
    compiling 'loss' = 'crossentropy'
"""



"""
PART 3: Training the ANN
"""

"""~~~~Compiling the ANN~~~~"""
ANN.compile(optimizer='adam', loss= 'binary_crossentropy', metrics= ['accuracy'])


"""~~~~Training the ANN on the Training set~~~~"""
ANN.fit(X_train, y_train, batch_size= 32, epochs = 100)



"""
PART 4: Predict a Result
"""


"""
Credit Score: 600       Country: France
Gender: Male            Age: 40
Tenure: 3               Balance:60000
Num. of Products: 2     Credit Card: Yes
Active member: Yes      Est. Salary: 50000
"""
answer = ANN.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))
#input of the predict method is a 2D array so brackets with brackets inside

#in this case: [three-digit-variable of Geography code, CreditScore, Gender code, Age, ... , Estimated Salary]
#if result close to 0: the customer will not leave the bank

#make sure to apply the scaling used during the training
#use 'transform' to use the same settings as previously set
#and 'fit_transform' to refit the weights

"""From probability to True/False"""
print (
       "\nIs the customer going to leave the bank?: ",
       answer[0,0] > 0.5)


"""~~~~Predicting the test set result~~~~"""
y_pred = ANN.predict(X_test) #use the ANN on the test sample
y_pred = (y_pred > 0.5) #get true/false readings instead of probabilities

y_test = np.array(y_test) #convert answer sample to array
y_test = (y_test.reshape(len(y_test),1)) > 0.5 #convert answer sample to 2D array with true/false format
#.reshape('rows','columns')

test_results = np.concatenate((y_pred,y_test), axis=1) #concatenate predictions and real results in a 2D array
# array([[pred, real],...])

print ("\n",test_results)

"""~~~~Making the Confusion Matrix~~~~"""
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print ("\nConfusion Matrix\n",cm)
# [[Correct that they stayed(00), Incorrect saying that they left(10)]
#    [Incorrect saying that they stayed(01), Correct that they left(11)]]
print ("Accuracy\n",accuracy_score(y_test, y_pred))

"""Replicate 'confusion_matrix' and 'accuracy_score' """
SS = LL = SL = LS = 0
for i in test_results:
    if (i[0] == 0) & (i[1] == 0): 
        SS+=1
    elif (i[0] == 0) & (i[1] == 1):
        SL+=1
    elif (i[0] == 1) & (i[1] == 1):
        LL+=1
    elif (i[0] == 1) & (i[1] == 0):
        LS+=1
        
print (
        "\n [[%d  %d]\n [ %d  %d]]"% (SS,LS,SL,LL),
        "\nGrade: %d%%" % ((SS+LL)/len(test_results)*100),
        "\nTotal: ", len(test_results),
        "\nPrediction | Real Result",
        "\nStayed | Stayed: ", SS,
        "\nLeft | Stayed: ", LS,
        "\nStayed | Left: ", SL,
        "\nLeft | Left: ", LL,
        )