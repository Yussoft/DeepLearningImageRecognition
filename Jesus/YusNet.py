from keras.models import Sequential
from keras.layers.convolutional import Conv2D # Class responsible for performing convolution
from keras.layers.convolutional import MaxPooling2D # Allows max-pooling operations
from keras.layers.core import Activation # For applying a particular activation function
from keras.layers.core import Flatten # To flatten our network topology into fully-connected
from keras.layers.core import Dense
from keras.layers import Dropout
from keras import backend as K
import keras.preprocessing.image

import os as os # Move to directories

def build_YusNet():
    """Function used to create the model.
    width: width of the input images.
    height: height of the input images.
    depth: the number of channels in our dataset.
    classes: classes to recognize.
    """

    # Sequential model is chosen since we will be sequentially adding layers to 
    # the model
    model = Sequential() 
    input_shape = (128, 128, 3) # Default ordering for tensorflow

    # Now that the model has been set, we will add layers to it. 
    # Convolutional layer -> ReLU layer -> Pooling Layer

    model.add(Conv2D(32, (12, 12), padding="same", input_shape = input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

    model.add(Conv2D(64, (6, 6), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(320))
    model.add(Activation("relu"))

    # softmax classifier, this layer has as many fully connected neurons as 
    # values there is in the class to be recognized.
    model.add(Dense(2)) # 2 Classes
    model.add(Activation("softmax"))

    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=["accuracy"])
    # return the constructed network architecture
    return model