# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 12:07:08 2022

@author: ABHRANIL
"""

import tensorflow as tf
import tensorflow.keras.layers as L

tf.keras.backend.clear_session()

def create_model():
    layers = [L.Input(shape = (28, 28, 3)), 
              L.SeparableConv2D(filters = 64, kernel_size = (1, 1), activation = 'relu'), 
              L.MaxPool2D(pool_size = (2, 2)),
              L.SeparableConv2D(filters = 128, kernel_size = (1, 1), activation = 'relu'),
              L.MaxPool2D(pool_size = (2, 2)),
              L.SeparableConv2D(filters = 256, kernel_size = (1, 1), activation = 'relu'),
              L.GlobalMaxPool2D(),
              L.Dense(256),
              L.ReLU(),
              L.Dense(64),
              L.ReLU(),
              L.Dense(26, activation = 'softmax')]
    

    model = tf.keras.Sequential(layers = layers)

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

model = create_model()

print(model.summary())