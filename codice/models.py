"""few models. first ones quite naive.
cnn_model has great potential. with this partition btw train and test
got test accuracy = 0.967 :))"""

import os
import sys
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Rescaling, Input, MaxPool2D, BatchNormalization, Activation
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np
from PIL import Image
from utils import plot, callbacks, read_imgs
from sklearn.utils import shuffle

""" 
used to convert dataset to 'png' estension (supported by keras.utils.image_dataset_from_directory)   
for file in os.listdir('datasets/Mammography_micro/Test/0'):
    filename, extension  = os.path.splitext(file)
    if extension == ".pgm":
        new_file = "{}.png".format(filename)
        with Image.open(os.path.join('datasets/Mammography_micro/Test/0',file)) as im:
            im.save(new_file)
 """

batch_size = 64
img_height = 60
img_width = 60
split = 0.4


def set_hyperp(args):
    """Create a dictionary containg the user-selected hps. It is set to be a 
    global variable so that it is accessible to the hypermodel aswell."""
    global hyperp
    hyperp  = {
            'depth' : args.net_depth,
            'Conv2d_init': args.Conv2d_init,
            'dropout' : args.dropout_rate

    }

def hyp_tuning_model(hp):
    shape = (60, 60, 1)
    model = Sequential()
    model.add(Input(shape=shape))
    hp_Dense_units = 256


    hp_depth = hp.Choice('depth', hyperp['depth'])
    hp_Conv2D_init = hp.Choice('Conv2d_init', hyperp['Conv2d_init'])
    hp_dropout = hp.Choice('dropout', hyperp['dropout'])

    model.add(Conv2D(hp_Conv2D_init, (3, 3), activation='relu', padding='same', strides=1, name='conv_1', input_shape=shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides = 2, name='maxpool_1'))
    model.add(Dropout(hp_dropout))

    model.add(Conv2D(2*hp_Conv2D_init, (3, 3), activation='relu', padding='same', strides=1, name='conv_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2,  name='maxpool_2'))
    model.add(Dropout(hp_dropout))

    model.add(Conv2D(4*hp_Conv2D_init, (3, 3), activation='relu', padding='same', strides=1, name='conv_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2, name='maxpool_3'))
    model.add(Dropout(hp_dropout))

    model.add(Flatten())
    model.add(Dropout(hp_dropout + 0.1))

    for i in range(hp_depth):
        units = hp_Dense_units/(i+1)
        model.add(Dense(units, activation='relu', name=f'dense_{i}'))

    model.add(Dense(1, activation='sigmoid', name='output'))

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    pass