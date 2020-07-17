#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
np.set_printoptions(suppress=True)
import tensorflow as tf
import pandas as pd


filePath = os.path.join(sys.path[0],'Churn_Modelling.csv')
try:    dataset = pd.read_csv(filePath)
except FileNotFoundError: pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:-1]
Y = dataset.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder as le
X['Gender'] = le().fit_transform(X['Gender'])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler as sc
X_train = sc().fit_transform(X_train)
X_test = sc().fit_transform(X_test)


ANN = tf.keras.models.Sequential()

ANN.add(tf.keras.layers.Dense(units=6, activation='relu'))
ANN.add(tf.keras.layers.Dense(units=6, activation='relu'))
ANN.add(tf.keras.layers.Dense(units=6, activation='relu'))
ANN.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ANN.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

ANN.fit(X_train, y_train, batch_size=64, epochs= 10)

y_pred = ANN.predict(X_test)
y_pred = (y_pred > 0.5)

y_test = np.array(y_test)
y_test = (y_test.reshape(len(y_test),1)) > 0.5

from sklearn.metrics import confusion_matrix, accuracy_score
print (confusion_matrix(y_test, y_pred))
print ("Accuracy\n",accuracy_score(y_test, y_pred))