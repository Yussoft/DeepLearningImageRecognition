# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from YusNet import build_YusNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import pandas as pandas

# Initialize the data and labels
print("[INFO] Loading images...")
data = []
labels = []

# Pistol max.:   300x300px
# Pistol min.:  160x120px
# Smartphone max:  1882x1919px
# Smartphone min:  49x51px
directory = "C:\\Users\\Yus\\Desktop\\DeepLearningImageRecognition\\Train"

image_paths = sorted(list(paths.list_images(directory)))
random.seed(77189383)
random.shuffle(image_paths)

height = 128
width = 128
for image_path in image_paths:

    # load the image, pre-process it, and store it in the data list
    if(image_path[-4:]==".jpg" or image_path[-4:]==".png" or image_path[-4:]==".JPG"):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (height, width))
        image = img_to_array(image)
        data.append(image)
    
        # extract the class label from the image path and update the
        # labels list
        label = image_path.split(os.path.sep)[-2]
        label = 1 if label == "Pistol" else 0
        labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
 
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size = 0.25, random_state = 77183983)
 
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes = 2)
testY = to_categorical(testY, num_classes = 2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range = 45, width_shift_range = 0.1,
	height_shift_range=0.2, shear_range=0.2, zoom_range=0.25,
	horizontal_flip=True, fill_mode = "nearest")

# initialize the model
print("[INFO] compiling model...")

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 300
INIT_LR = 1e-3
BS = 64
# build_CNN(height, width, f1, f2, kernel_size, pool_size, stride, neurons)
model = build_YusNet()
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
 
# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size = BS),
	validation_data = (testX, testY), steps_per_epoch = len(trainX) // BS,
	epochs=EPOCHS, verbose = 1)
 
# save the model to disk
print("[INFO] serializing network...")
model.save("modelo_yusnet_1")

# Save the best evaluation accuracy
best_acc = 0
best_epoch = 0
val_acc = H.history["val_acc"]

for i in range(len(val_acc)):
    if val_acc[i] > best_acc:
        best_acc = val_acc[i]
        best_epoch = i

print("Best val_acc: ",best_acc)
