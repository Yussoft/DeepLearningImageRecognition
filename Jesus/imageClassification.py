from keras.models import Sequential
from keras.layers.convolutional import Conv2D # Class responsible for performing convolution
from keras.layers.convolutional import MaxPooling2D # Allows max-pooling operations
from keras.layers.core import Activation # For applying a particular activation function
from keras.layers.core import Flatten # To flatten our network topology into fully-connected
from keras.layers.core import Dense
from keras import backend as K
import keras.preprocessing.image

import os as os# move to directories

def build_CNN(width, height, depth, classes):
    """Function used to create the model.
    width: width of the input images.
    height: height of the input images.
    depth: the number of channels in our dataset.
    classes: classes to recognize.
    """
    # Sequential model is chosen since we will be sequentially adding layers to 
    # the model
    model = Sequential() 
    input_shape = (height, width, depth) # Default ordering for tensorflow

    # Now that the model has been set, we will add layers to it. 
    # Convolutional layer -> ReLU layer -> Pooling Layer

    # 20 convolutional filters of 5x5
    model.add(Conv2D(20, (5, 5), padding="same", input_shape = inputShape))
    model.add(Activation("relu"))
    # Max-pooling of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second set of layers 

    # 50 convolutional filters rather than 20, the deeper we go into the net, 
    # convolutional filters increase
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

# Load the images and transform them into matrixes
directory = "C:\\Users\\Yus\\Desktop\\DeepLearningImageRecognition\Jesus"
os.chdir(directory)

print(os.getcwd())