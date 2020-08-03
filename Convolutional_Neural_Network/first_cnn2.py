#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 9:29:04 2020

@author: djinnalexio

Convolutional Neural Network
"""
import os, sys
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

"Preprocessing"
def preprocessing(filePathPrefix=''):
    train_datagen = image.ImageDataGenerator(
        rescale=1./255, zoom_range=.2, shear_range=.2, horizontal_flip=True)

    training_set = train_datagen.flow_from_directory(
        (filePathPrefix+'dataset/training_set'),
        target_size=(64,64), batch_size=32, class_mode='binary')

    test_datagen=image.ImageDataGenerator(rescale=1./255)

    test_set= test_datagen.flow_from_directory(
        (filePathPrefix+'dataset/test_set'),
        target_size=(64, 64), batch_size=32, class_mode='binary')

    return training_set, test_set


try: training_set, test_set = preprocessing()
except FileNotFoundError:
    training_set, test_set = preprocessing(filePathPrefix=sys.path[0]+'/')

"Building the ANN"
CNN = tf.keras.models.Sequential()

CNN.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                               activation='relu', input_shape=[64, 64, 3]))

CNN.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

CNN.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                               activation='relu'))

CNN.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

CNN.add(tf.keras.layers.Flatten())

CNN.add(tf.keras.layers.Dense(units=128, activation='relu'))

CNN.add(tf.keras.layers.Dense(units=128, activation='relu'))

CNN.add(tf.keras.layers.Dense(units=128, activation='relu'))

CNN.add(tf.keras.layers.Dense(units=128, activation='relu'))

CNN.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"Training the ANN"
CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

CNN.fit(x=training_set, validation_data=test_set, epochs=30)


"Single Prediction"
filePath = 'dataset/single_prediction/cat_or_dog_1.jpg'
try:
    test_image = image.load_img(filePath, target_size=(64, 64))
    
except FileNotFoundError:
    test_image = image.load_img(os.path.join(
        sys.path[0], filePath), target_size=(64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = CNN.predict(test_image)

if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"

print("This is the image of a %s" % prediction)
