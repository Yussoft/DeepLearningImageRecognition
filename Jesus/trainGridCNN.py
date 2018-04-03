from keras.models import Sequential
from keras.layers.convolutional import Conv2D # Class responsible for performing convolution
from keras.layers.convolutional import MaxPooling2D # Allows max-pooling operations
from keras.layers.core import Activation # For applying a particular activation function
from keras.layers.core import Flatten # To flatten our network topology into fully-connected
from keras.layers.core import Dense
from keras import backend as K
import keras.preprocessing.image
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os



# Initialize the data and labels
print("[INFO] Loading images...")
data = []
labels = []

# Pistol max.:  300x300px
# Pistol min.:  160x120px
# Smartphone max:  1882x1919px
# Smartphone min:  49x51px
directory = "C:\\Users\\Yus\\Desktop\\DeepLearningImageRecognition\\Train"

image_paths = sorted(list(paths.list_images(directory)))
np.random.seed(77189383)
np.random.shuffle(image_paths)

for image_path in image_paths:
    # The last 4 characters of the paths have to be .jpg / .png / .JPG
    if(image_path[-4:]==".jpg" or image_path[-4:]==".png" or image_path[-4:]==".JPG"):
        # Read, resize and transform the images to array
        image = cv2.imread(image_path)
        image = cv2.resize(image, (60, 60))
        image = img_to_array(image)
        data.append(image)
        # Extract the class label from the image path (Pistol/Smartphone)
        label = image_path.split(os.path.sep)[-2]
        label = 1 if label == "Pistol" else 0
        labels.append(label)

# Scale the raw pixel intensities [0, 255] to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# model = build_CNN(width = width, height = height, depth = 3, classes = 2)
model = KerasClassifier(build_fn = build_CNN, verbose = 0)

batch_size = [10]
epochs = [10]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(data, labels)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))