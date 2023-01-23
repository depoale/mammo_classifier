import os
import sys
import keras_tuner as kt
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Rescaling, Input, MaxPool2D, BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np
from PIL import Image
from utils import get_data, read_imgs, plot, callbacks
import sklearn
from sklearn.model_selection import train_test_split


train_path = r"c:\\Users\\franc\Desktop\DISPENSE 4 ANNO\\Computing Methods\\ESAME\\cmepda_prj\data_png\\Train"
test_path = r"c:\\Users\\franc\Desktop\DISPENSE 4 ANNO\\Computing Methods\\ESAME\\cmepda_prj\data_png\\Test"

batch_size = 64
img_height = 60
img_width = 60
split = 0.3
random_st = 42

def hyp_tuning_model(hp):
    shape = (60, 60, 1)
    model = Sequential()

    hp_learning_rate = hp.Choice('learning_rate', values=[5e-2, 1e-2, 1e-3])
    hp_depth = hp.Int('depth', min_value = 2, max_value = 4, step=1)
    hp_Dense_units = hp.Choice('Dense_units', values=[128, 256])
    
    
    model.add(Rescaling(scale=1./255.))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1', input_shape=shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), name='maxpool_1'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), name='maxpool_2'))
    model.add(Dropout(0.01))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), name='maxpool_3'))
    model.add(Dropout(0.01))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), name='maxpool_4'))

    model.add(Flatten())
    model.add(Dropout(0.1))

    for i in range(hp_depth):
        model.add(Dense(hp_Dense_units, activation='relu', name=f'dense_{i}'))

    model.add(Dense(1, activation='sigmoid', name='output'))

    model.compile(loss='MSE', optimizer= Adam(learning_rate = hp_learning_rate), metrics=['accuracy'])
    
    return model

'''
def cnn_model(shape=(60, 60, 1), learning_rate=1e-3, verbose=False):
    
    CNN for microcalcification clusters classification.

    Parameters
    ----------
    shape : tuple, optional
        The first parameter.
    verbose : bool, optional
        Enables the printing of the summary. Defaults to False.

    Returns
    -------
    model
        Return the convolutional neural network.
    """
    model = Sequential()
    model.add(Rescaling(scale=1./255.))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1', input_shape=shape))
    model.add(MaxPooling2D((2, 2), name='maxpool_1'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
    model.add(MaxPooling2D((2, 2), name='maxpool_2'))
    model.add(Dropout(0.01))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
    model.add(MaxPooling2D((2, 2), name='maxpool_3'))
    model.add(Dropout(0.01))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
    model.add(MaxPooling2D((2, 2), name='maxpool_4'))

    model.add(Flatten())
    model.add(Dropout(0.1))

    model.add(Dense(256, activation='relu', name='dense_2'))
    model.add(Dense(128, activation='relu', name='dense_3'))
    model.add(Dense(1, activation='sigmoid', name='output'))

    model.compile(loss='MSE', optimizer= Adam(learning_rate=learning_rate), metrics=['accuracy'])
    
    if verbose:
      model.summary()
  
    return model
'''


if __name__ == '__main__':
    #x_train, y_train = read_imgs(train_path, [0, 1])
    #x_test, y_test = read_imgs(test_path, [0, 1]) 
    #X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.30, random_state = random_st)

    train, val, test = get_data(train_path=train_path, test_path=test_path)
  
    tuner = kt.BayesianOptimization(hyp_tuning_model, objective='val_loss', max_trials=5, overwrite=True, directory='tuner')
    tuner.search(train, epochs=5, validation_data=val)
    
    tuner.results_summary()

    best_model = tuner.get_best_models()[0]
    history = best_model.fit(train, batch_size=batch_size , epochs=50, validation_data=val, callbacks=callbacks)
    
    #model = cnn_model()

    #history = model.fit(train, batch_size=batch_size , epochs=100, validation_data=val, callbacks=callbacks)
 












    
