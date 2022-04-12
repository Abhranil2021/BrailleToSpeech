# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:46:26 2022

@author: ABHRANIL
"""

import os
from shutil import copyfile

# making a directory of images in which images of a particular character will be stored in the same folder
image_dir = "C:/Users/ABHRANIL/OneDrive/Documents/Projects/BrailleToSpeech/Dataset/Braille Dataset/Images/"
os.mkdir(image_dir)
alpha = 'a'
for i in range(0, 26): 
    os.mkdir(image_dir + alpha)
    alpha = chr(ord(alpha) + 1)
    
root_dir = "C:/Users/ABHRANIL/OneDrive/Documents/Projects/BrailleToSpeech/Dataset/Braille Dataset/Braille Dataset/"
for file in os.listdir(root_dir):
    letter = file[0]
    copyfile(root_dir + file, image_dir + letter + '/' + file)