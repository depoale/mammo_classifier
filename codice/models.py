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


train_path= os.path.join(os.getcwd(),'data_png_WAVELET' ,'Train')
test_path=os.path.join(os.getcwd(),'data_png_WAVELET' ,'Test')

def cnn_classifier(shape=(60, 60, 1), verbose=False):
    """removed resizing layer
    """

    model = Sequential()
    model.add(Input(shape=shape))
    model.add(Conv2D(32, (3, 3),
    activation='relu',
    padding='same',
    name='conv_1',
    input_shape=shape)
    )
    model.add(MaxPooling2D((2, 2), name='maxpool_1'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), name='maxpool_2'))
    model.add(Dropout(0.05))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), name='maxpool_3'))
    model.add(Dropout(0.05))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), name='maxpool_4'))

    model.add(Flatten())
    model.add(Dropout(0.1))

    model.add(Dense(256, activation='relu', name='dense_2'))
    model.add(Dense(128, activation='relu', name='dense_3'))
    model.add(Dense(1, activation='sigmoid', name='output'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=5e-2), metrics=['accuracy'])

    if verbose:
        model.summary()

    return model


def make_model(shape=(60, 60, 1)):
  model = Sequential([
      
      Conv2D(8, (3,3), padding='same', input_shape=shape),
      BatchNormalization(),
      Activation('relu'),

      MaxPool2D((2,2), strides=2),
      #Dropout(0.4),

      Conv2D(16, (3,3), padding='same'),
      BatchNormalization(),
      Activation('relu'),

      MaxPool2D((2,2), strides=2),
      #Dropout(0.4),

      Conv2D(32, (3,3), padding='same'),
      BatchNormalization(),
      Activation('relu'),

      MaxPool2D((2,2), strides=2),
      #Dropout(0.4),

      Flatten(),    #Flatten serves as a connection between the convolution and dense layers.

      Dense(10, activation='relu'),
      Dropout(0.2),
      Dense(1, activation='sigmoid')
     
  ])
  model.compile(loss='binary_crossentropy', optimizer= Adam(learning_rate = 0.001), metrics=['accuracy'])
  return model


def hyp_tuning_model(hp):
    shape = (60, 60, 1)
    model = Sequential()
    model.add(Input(shape=shape))

    hp_depth = hp.Int('depth', min_value = 1, max_value = 3, step=1)
    hp_Dense_units = hp.Choice('Dense_units', values=[256])
    hp_Conv2D_init = hp.Choice('Conv2d_init', values=[10, 20, 30])
    hp_dropout = hp.Choice('dropout', values=[0.0, 0.05])
    hp_Conv2d_size = hp.Choice('Conv2D_size', values=[3, 5])
    

    model.add(Conv2D(hp_Conv2D_init, (3, 3), activation='relu', padding='same', strides=1, name='conv_1', input_shape=shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides = 2, name='maxpool_1'))
    model.add(Dropout(hp_dropout))

    model.add(Conv2D(2*hp_Conv2D_init, (3, 3), activation='relu', padding='same', strides=1, name='conv_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2,  name='maxpool_2'))
    model.add(Dropout(hp_dropout))

    model.add(Conv2D(4*hp_Conv2D_init, (hp_Conv2d_size, hp_Conv2d_size), activation='relu', padding='same', strides=1, name='conv_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2, name='maxpool_3'))
    model.add(Dropout(hp_dropout))

    """ model.add(Conv2D(128, (3, 3), activation='relu', padding='zero', name='conv_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), name='maxpool_4')) """

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