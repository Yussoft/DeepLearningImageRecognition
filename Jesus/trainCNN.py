# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# With all the imports we can:
# Load our image dataset from disk
# Pre-process the images
# Instantiate our Convolutional Neural Network
# Train our image classifier

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
 
# initialize the data and labels
print("[INFO] Loading images...")
data = []
labels = []

# Go the the train images directory and load them PISTOLS!
os.chdir("C:\\Users\\Yus\\Desktop\\DeepLearningImageRecognition\\Train\\Pistol")
file_names = os.listdir()

print(file_names)
pistol_max_width = 0
pistol_min_width = 0
pistol_max_height = 0
pistol_min_height = 0

for im_name in file_names:

    file_name = os.path.splitext(im_name)
    file_extension = file_name[1]
    
    if file_extension == ".jpg":
        n_image = cv2.imread(im_name)
        shape = n_image.shape

        print(shape[0])

        if shape[0] > pistol_max_width:
            pistol_max_width = shape[0]

        elif shape[0] < pistol_min_width:
            pistol_min_width = shape[0]
        
        if shape[1] > pistol_max_height:
            pistol_max_height = shape[1]
        
        elif shape[1] < pistol_min_height:
            pistol_min_height = shape[1]
       

print("Pistol max height:",pistol_max_height)

print("Pistol min height:",pistol_min_height)

print(pistol_max_width)

print(pistol_min_width)