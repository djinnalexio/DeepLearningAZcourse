#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:53:27 2020

@author: djinnalexio

Convolutional Neural Network
"""

import sys, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
#was not able to download the keras module on Arch
#So, use the library under tensorflow
print ("Tensorflow version: %s" % tf.__version__)


"""~~~Image Preprocessing~~~"""

# Creating a function here to easily handle error in case complete file path is necessary
def Preprocessing(filePathPrefix=''):
    """Image Augmentation"""
    # transform the images of the training set to increase variety so that the CNN is not over-fitted on the raw images
    
    #the object that will apply transformation to the images in the training set
    train_datagen = image.ImageDataGenerator(
        rescale=1./255, #feature scaling, pixel values range from 0 to 255, so divide than by 255 to get values between 0 and 1
        shear_range=0.2, #other variables are transformations applied to the images
        zoom_range=0.2,
        horizontal_flip=True)
    
    training_set = train_datagen.flow_from_directory( #will connect the object to the directory containing the images
            (filePathPrefix+'dataset/training_set'), #the folder to be used
            target_size=(64, 64), #the final image size, the small = the faster the training
            batch_size=32,
            class_mode='binary')
    
    
    """Preprocessing of the test set"""
    #we do not want to transform images of the test set so we will only apply feature scaling
    test_datagen = image.ImageDataGenerator(rescale=1./255)
    
    test_set = test_datagen.flow_from_directory( #will connect the object to the directory containing the images
            (filePathPrefix+'dataset/test_set'),
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')
    
    return training_set, test_set

try: 
    training_set, test_set = Preprocessing()
    print ("file obtained from path from filename")
except FileNotFoundError:
    training_set, test_set = Preprocessing(filePathPrefix = sys.path[0] + '/') #adding '/' instead of using 'os.path.join'
    print ("file obtained from absolute path")


"""~~~Building the CNN~~~"""

CNN = tf.keras.models.Sequential()

"Convolution Layer" # adding the layer to obtain a feature map
CNN.add(tf.keras.layers.Conv2D(
    filters= 64, #how many feature detectors we want
    kernel_size= 3, #the size of the filter/kernel (x*x)
    activation='relu',
    input_shape=[64,64,3]) #the shape of the input: in this case, 64x64 pixels and 3 for RGB values (if using black/white, then we only need 1)
    )


"Pooling Layer" # adding the layer to obtain a pooled feature map
CNN.add(tf.keras.layers.MaxPool2D(
    pool_size=(2,2), #dimensions of the pool kernel
    strides= 2, #the size of each step
    padding='valid' #how the empty pixels when the kernel reaches the end the line are handled
#'valid' = ignores the missing pixels | 'same' = add fke pixels that are equal to 0
    ))
    
    
"Second Convolution+Pooling Layer" # adding the layer to obtain a feature map
CNN.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= 3, activation='relu')) #the input_size is only necessary for the first layer
CNN.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides= 2))


"Flattening"
CNN.add(tf.keras.layers.Flatten()) #turns the matrices into lists of inputs

"Full Connection"
CNN.add(tf.keras.layers.Dense(units=256, activation='relu'))
CNN.add(tf.keras.layers.Dense(units=256, activation='relu'))
CNN.add(tf.keras.layers.Dense(units=256, activation='relu'))


"Output Layer"
CNN.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



"""~~~Training the CNN~~~"""

"Compiling the CNN"
CNN.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

"Training the CNN"
CNN.fit(x = training_set, validation_data= test_set, epochs= 30)
#for this specific application, training and testing will be done in the same step



"""~~~Making a single prediction~~~"""

"Importing the image" #use try/except again to handle filenotfounderror
filePath = 'dataset/single_prediction/cat_or_dog_2.jpg'
try: test_image = image.load_img(
    filePath, #the variable of the test image
    target_size = (64,64)) #dimensions of the test image

except FileNotFoundError: test_image = image.load_img(
    os.path.join(sys.path[0],filePath),
    target_size = (64,64))

"converting from PIL to 2D array"
test_image = image.img_to_array(test_image)

"Adding an extra dimension for the batch"
#the CNN was trained on batches, therefore, even to make a single prediction, that test needs to be in a batch
test_image = np.expand_dims(test_image, axis = 0) #add a dimension at the beginning

"Making a prediction"
result = CNN.predict(test_image)
training_set.class_indices # indicates the values of each class (which class is 0, which class is 1)

if result[0][0] == 1: #there is only one item in the 3D array so the first 2 axis are at 0
    prediction = "dog"
else: prediction = "cat"

print("This is the image of a %s" % prediction)
