import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, InputLayer, Dropout, Rescaling, Flatten, Reshape, concatenate
import keras
import h5py
from sklearn.utils import shuffle
import keras_tuner as kt
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import statistics as stats
import time
from utils import read_imgs
from kfold import fold_tuner_auto
PATH = 'total_data'


def get_autoenc():
    shape_in = (60,60,1)
    model = keras.models.Sequential()
    model.add(InputLayer(shape_in))
    model.add(Rescaling(scale=1./255.))
    model.add(Flatten())
    model.add(Dropout(0.05))
    model.add(Dense(256, activation='relu', name='dense1'))
    model.add(Dropout(0.05))
    model.add(Dense(160, activation='relu', name='dense2'))
    model.add(Dropout(0.05))
    model.add(Dense(64, activation='relu', name='enc'))  #neck of the bottle neck
    model.add(Dropout(0.05))
    model.add(Dense(160, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.05))

    
    model.add(Dense(np.prod(shape_in)))
    model.add(Reshape(shape_in))
    model.compile(loss='MSE', optimizer=Adam(learning_rate=0.01))
    return model

def get_encoder():
    shape_in = (60,60,1)
    inputs = keras.layers.Input(shape_in)
    x = Rescaling(scale=1./255.)(inputs)
    x = Flatten()(x)
    x=Dropout(0.05)(x)
    x = Dense(256, activation='relu', trainable=False, name='dense1')(x)
    x=Dropout(0.05)(x)
    x = Dense(160, activation='relu', trainable=False, name='dense2')(x)
    x=Dropout(0.05)(x)
    enc = Dense(64, activation='relu', trainable=False, name='enc')(x)
    model = keras.Model(inputs, enc)
    model.compile(loss='MSE', optimizer=Adam(learning_rate=0.01))
    return model

def get_decoder():
    pass



def get_concatenated_deepNN():
    shape_in = (60,60,1)
    inputs = keras.layers.Input(shape_in)
    x = Rescaling(scale=1./255.)(inputs)
    x = Flatten()(x)
    x=Dropout(0.05)(x)
    x = Dense(256, activation='relu', trainable=False, name='dense1')(x)
    x=Dropout(0.05)(x)
    x = Dense(160, activation='relu', trainable=False, name='dense2')(x)
    x=Dropout(0.05)(x)
    enc = Dense(64, activation='relu', trainable=False, name='enc')(x)

    #trainable net
    x = Dense(256, activation='relu')(enc)
    x= Dropout(0.05)(x)
    x = Dense(256, activation='relu', name='dense1.1')(x)
    x= Dropout(0.05)(x)
    x = Dense(128, activation='relu',name='dense1.2')(x)
    x= Dropout(0.05)(x)
    x = Dense(64, activation='relu', name='dense1.3')(x)
    outputs = Dense(1, activation='sigmoid', name='outputs')(x)
    model =keras.Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer= Adam(learning_rate=0.01), metrics=['accuracy'])
    return model

if __name__=='__main__':
    X, y = read_imgs(PATH, [0,1])
    X, y = shuffle(X, y)
    X_train, X_test = X[:700], X[700:]
    y_train, y_test = y[:700], y[700:]
    autoencoder = get_autoenc()
    history = autoencoder.fit(X_train, X_train, epochs=100, validation_split=0.2, 
                callbacks= [EarlyStopping(monitor='val_loss', min_delta=5e-3, patience=20, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.25, min_delta=1e-4,patience=10, verbose=1)])

    autoencoder.save_weights("weights_auto.h5")

    #lerning curve plot
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(loss)+1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show(block=False)

   
    print(f'test loss: {autoencoder.evaluate(X_test, y_test)}')
    new_model = get_concatenated_deepNN()
    new_model.load_weights("weights_auto.h5", by_name=True, skip_mismatch=True)
    new_model.summary()
    history = new_model.fit(X_train, y_train, epochs=100, validation_split=0.2, 
                callbacks= [EarlyStopping(monitor='val_accuracy', min_delta=5e-3, patience=20, verbose=1),
                ReduceLROnPlateau(monitor='val_accuracy', factor=0.25, min_delta=1e-4,patience=10, verbose=1)])
    print(f'test loss: {new_model.evaluate(X_test, y_test)}')


    

    




