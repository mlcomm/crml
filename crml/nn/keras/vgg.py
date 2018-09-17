import os,random
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import random, sys, keras

def vgg(in_shp = [2, 128], classes = ['busy' ,'idle']):
    K.set_image_dim_ordering('th')
    dr = 0.5
    model = models.Sequential()
    model.add(Reshape([1]+in_shp, input_shape=in_shp))
    
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(64, 2, 9, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform')) 
    model.add(ZeroPadding2D((0, 2)))
    model.add(MaxPooling2D((1,2),strides=(1,2)))
    
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(64, 1, 9, border_mode='valid', activation="relu", name="conv2", init='glorot_uniform')) 
    model.add(ZeroPadding2D((0, 2)))
    model.add(MaxPooling2D((1,2),strides=(1,2))) 
    
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(64, 1, 9, border_mode='valid', activation="relu", name="conv3", init='glorot_uniform')) 
    model.add(ZeroPadding2D((0, 2)))
    model.add(MaxPooling2D((1,2),strides=(1,2))) 
   
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(64, 1, 9, border_mode='valid', activation="relu", name="conv4", init='glorot_uniform')) 
    model.add(ZeroPadding2D((0, 2)))
    model.add(MaxPooling2D((1,2),strides=(1,2))) 
   
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(64, 1, 9, border_mode='valid', activation="relu", name="conv5", init='glorot_uniform')) 
    model.add(ZeroPadding2D((0, 2)))
    model.add(MaxPooling2D((1,2),strides=(1,2))) 
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
    model.add(Dropout(dr))
    
    model.add(Dense(128, activation='relu', init='he_normal', name="dense2"))
    model.add(Dropout(dr))
    
    model.add(Dense(len(classes), init='he_normal', name="dense3"))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model
