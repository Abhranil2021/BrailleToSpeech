# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:59:19 2022

@author: ABHRANIL
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from ModelComponents.Model import create_model

image_dir = "C:/Users/ABHRANIL/OneDrive/Documents/Projects/BrailleToSpeech/Dataset/Braille Dataset/Images/"


train_data = image_dataset_from_directory(image_dir, 
                                          label_mode = 'categorical',
                                          image_size = (28,28), 
                                          validation_split = 0.25, 
                                          subset = 'training',
                                          seed = 50)
val_data = image_dataset_from_directory(image_dir,
                                        label_mode = 'categorical',
                                        image_size = (28,28), 
                                        validation_split = 0.25, 
                                        subset = 'validation',
                                        seed = 50)

weights_dir = "C:/Users/ABHRANIL/OneDrive/Documents/Projects/BrailleToSpeech/Model/Weights"

model_ckpt = ModelCheckpoint(weights_dir, save_best_only = True)
reduce_lr = ReduceLROnPlateau(patience = 8)
early_stop = EarlyStopping(patience = 15)

model = create_model()

history = model.fit(train_data,
                    validation_data = val_data,
                    epochs = 200,
                    callbacks = [model_ckpt, reduce_lr])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(200)

plt.figure(figsize = (8, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label ='Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')

plt.show()
