import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Dropout, Flatten, Reshape
import keras
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras import Model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import keras_tuner as kt
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import statistics as stats
import time
from utils import read_imgs
from kfold import fold_tuner_auto
PATH = 'total_data'


def get_autoenc(hp):
    shape_in = (60,60,1)
    model = keras.models.Sequential()
    model.add(Input(shape=(shape_in,)))
    model.add(Flatten())
    
    hp_units = hp.Choice('units',[128, 256])
    hp_dropout = hp.Choice('dropout', [ 1e-2, 1e-3, 0.0])
    hp_depth = hp.Int('depth', min_value=4, max_value=8, step=2)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    
    for i in range(hp_depth+1):
        model.add(Dropout(hp_dropout))
        fun = lambda x: round(hp_units/10 +9*hp_units/10*(np.cos(np.pi*x/hp_depth))**2)
        model.add(Dense(fun(i), activation='relu'))
        
    model.add(Dense(shape_in))
    model.add(Reshape(shape_in))
    model.compile(loss='MSE', optimizer=Adam(learning_rate=hp_learning_rate))
    return model

def get_encoder(hp):
    shape_in = (60,60,1)
    model = keras.models.Sequential()
    model.add(Input(shape=(shape_in,)))
    model.add(Flatten())
    
    hp_units = hp.Choice('units',[64, 128, 256])
    hp_dropout = hp.Choice('dropout', [ 1e-2, 1e-3, 0.0])
    hp_depth = hp.Int('depth', min_value=4, max_value=8, step=2)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    
    for i in range(hp_depth+1):
        if i<=(hp_depth+1)//2:
            model.add(Dropout(hp_dropout))
            fun = lambda x: round(hp_units/10 +9*hp_units/10*(np.cos(np.pi*x/hp_depth))**2)
            model.add(Dense(fun(i), activation='relu',trainable=False))
    
    model.compile(loss='MSE', optimizer=Adam(learning_rate=hp_learning_rate))
    return model

if __name__=='__main__':
    X, y = read_imgs(PATH, [0,1])
    fold_tuner_auto(X, X, k=5, modelBuilder=get_autoenc)