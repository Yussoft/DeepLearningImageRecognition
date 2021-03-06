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

def build_CNN(parameters):
    """Function used to create the model.
    width: width of the input images.
    height: height of the input images.
    depth: the number of channels in our dataset.
    classes: classes to recognize.
    """
    height = parameters[0]
    width = parameters[1]
    f1 = parameters[2]
    f2 = parameters[3]
    kernel_size = parameters[4]
    pool_size = parameters[5]
    stride = parameters[6]
    neurons = parameters[7]

    # Sequential model is chosen since we will be sequentially adding layers to 
    # the model
    model = Sequential() 
    input_shape = (height, width, 3) # Default ordering for tensorflow

    # Now that the model has been set, we will add layers to it. 
    # Convolutional layer -> ReLU layer -> Pooling Layer

    # 20 convolutional filters of 5x5 size
    model.add(Conv2D(f1, (kernel_size, kernel_size), padding="same", input_shape = input_shape))
    model.add(Activation("relu"))
    # Max-pooling of 2x2
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(stride, stride)))

    # Second set of layers 

    # 50 convolutional filters rather than 20, the deeper we go into the net, 
    # convolutional filters increase.
    model.add(Conv2D(f2, (kernel_size, kernel_size), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(stride, stride)))

    model.add(Conv2D(75, (kernel_size, kernel_size), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(stride, stride)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(neurons))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))

    # softmax classifier, this layer has as many fully connected neurons as 
    # values there is in the class to be recognized.
    model.add(Dense(2)) # 2 Classes
    model.add(Activation("softmax"))

    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=["accuracy"])
    # return the constructed network architecture
    return model