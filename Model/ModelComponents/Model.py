# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 12:07:08 2022

@author: ABHRANIL
"""

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.regularizers import l2

tf.keras.backend.clear_session()

def create_model():
    layers = [L.Input(shape = (28, 28, 1)), 
              L.experimental.preprocessing.Rescaling(scale = 1.0/255),
              L.SeparableConv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'), 
              L.MaxPool2D(pool_size = (2, 2)),
              L.SeparableConv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'),
              L.MaxPool2D(pool_size = (2, 2)),
              L.SeparableConv2D(filters = 256, kernel_size = (2, 2), activation = 'relu'),
              L.GlobalMaxPool2D(),
              L.Dense(256),
              L.Dropout(0.5),
              L.ReLU(),
              L.Dense(64, kernel_regularizer = l2(2e-4)),
              L.Dropout(0.2),
              L.ReLU(),
              L.Dense(26, activation = 'softmax')]
    

    model = tf.keras.Sequential(layers = layers)

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

model = create_model()

print(model.summary())