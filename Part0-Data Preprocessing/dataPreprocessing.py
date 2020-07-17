#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:11:21 2020

@author: djinnalexio
"""
import os, sys
import numpy as np #allows to work with arrays
#import matplotlib.pyplot as plt #allows to plot charts
import pandas as pd #allows to import the dataset and create matrices


"""Importing Data"""
fileName = 'Data.csv'
try:    dataset = pd.read_csv(os.path.join(sys.path[0],fileName))
except FileNotFoundError:    dataset = pd.read_csv(fileName)

X = dataset.iloc[:, :-1].values #takes all columns but the last one
y = dataset.iloc[:,-1].values


"""Taking care of Missing Data"""
from sklearn.impute import SimpleImputer # 'SciKitLearn'
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #np.nan represents empty values, replace them with the mean
imputer.fit(X[:,1: ]) #fit will connect the imputer to the data to read the data and calculate the mean
X[:,1:3] = imputer.transform(X[:,1: ]) #transform will be making the replacements


"""Encoding Categorical data"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
#ColumnTransformer applies changes to columns of a set
#transformers= a list of ('name',the method, index of the column to transform)
X = np.array(ct.fit_transform(X))


"""Encoding the Dependent Variable"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


"""Splitting the dataset into Training set and Test set"""
#X_train = matrix of features for the training set    #X_test = matrix of features for the test set

#y_train = dependent variables of the training set    #y_test = dependent variables of the test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
#20% of observations will be dedicated to the test set
#we will have the same training set and test set


"""Feature Scaling"""
from sklearn.preprocessing import StandardScaler#keeps all numerical values in the same range
sc = StandardScaler()
X_train[:,3: ] = sc.fit_transform(X_train[:,3: ]) #do not include dummy variables while feature scaling
X_test[:,3: ] = sc.transform(X_test[:,3: ])

print(X_train)
print(X_test)