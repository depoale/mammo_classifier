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
from utils import read_imgs, get_rows_columns
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
    print(os.getcwd())
    for cl in classes:
        img_array, labels = read_imgs('total_data', cl)
        for idx in rnd_idx:
            ax=fig.add_subplot(rows, columns, i)
            ax.title.set_text(f'Label = {labels[idx]}')
            plt.imshow(img_array[idx], cmap='gray')
            i+=1
    plt.show()


'''def gCAM_plot(size, preds):
    fig = plt.figure(figsize=(8, 8))
    rows, columns = get_rows_columns(size)
    tmp = []
    fnames = glob.glob(os.path.join('gradCAM', '*.png'))
    tmp += [imread(fname) for fname in fnames]
    images = np.array(tmp, dtype='float32')[...]/255

    for i in range(size):
        ax = fig.add_subplot(rows, columns, i+1)
        ax.title.set_text(f'Label=1 Pred={np.rint(preds[i])}')
        plt.imshow(images[i])

    plt.show()'''


if __name__=='__main__':
    plottino()