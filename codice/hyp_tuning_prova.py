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
from utils import read_imgs, plot, callbacks
from kfold import fold, fold_tuner
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from models import hyp_tuning_model


PATH ='total_data'



if __name__ == '__main__':
    X, y = read_imgs(PATH, [0, 1])
    X, y = shuffle(X, y)
    fold_tuner(X, y, k=5, modelBuilder=hyp_tuning_model)











    
