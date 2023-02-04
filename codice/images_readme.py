import os
import sys
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Rescaling, Input
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, BinaryAccuracy
import time
import errno
from PIL import Image
import glob
import logging
from skimage.io import imread
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sn
import pandas as pd
import shutil
from utils import read_imgs
from keras.utils import image_dataset_from_directory

def plottino():
    
    rnd_idx = np.random.randint(0, 200, size = 3)
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 2
    classes = [[0],[1]]
    i=1
    for cl in classes:
        img_array, labels = read_imgs('wavelet_data', cl)
        for idx in rnd_idx:
            ax=fig.add_subplot(rows, columns, i)
            ax.title.set_text(f'Label = {labels[idx]}')
            plt.imshow(img_array[idx], cmap='gray')
            i+=1
    plt.show()

if __name__=='__main__':
    plottino()