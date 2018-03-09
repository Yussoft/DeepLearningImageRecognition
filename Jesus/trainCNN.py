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

for im_name in file_names:
    n_image = cv2.imread(im_name) # Read an imagep
    n_image = cv2.resize(n_image,(128,128)) # Resize it to 128x128pixels
    n_image = img_to_array(n_image) 
    data.append(n_image)

    label = "Pistol"
    labels.append(label)

print("[INFO] Training images (PISTOL) read:",len(data))

# Go the the train images directory and load them SMARTPHONES
os.chdir("C:\\Users\\Yus\\Desktop\\DeepLearningImageRecognition\\Train\\Smartphone")
file_names = os.listdir()

for im_name in file_names:
    n_image = cv2.imread(im_name) # Read an imagep
    n_image = cv2.resize(n_image,(128,128)) # Resize it to 128x128pixels
    n_image = img_to_array(n_image) 
    data.append(n_image)

    label = "Pistol"
    labels.append(label)

print("[INFO] Training images (SMARTPHONE) read:",len(data))    
print("[INFO] Training images (TOTAL) read:",len(data))

