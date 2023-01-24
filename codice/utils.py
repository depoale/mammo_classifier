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
from PIL import Image
import glob
import logging
from skimage.io import imread
import numpy as np

train_path = os.path.join(os.getcwd(),'data_png' ,'Train')
test_path = os.path.join(os.getcwd(),'data_png' ,'Test')

from keras.utils import image_dataset_from_directory

split=0.25

def read_imgs(dataset_path, classes):
    """Function reading all the images in a given folder which already contains
    two subfolder.

    Parameters
    ----------
        dataset_path : str
            Path to the image folder.
        classes : list
            0 and 1 mean normal tissue and microcalcification clusters, respectively.

    Returns
    -------
        array: numpy_array
            Array containing the value of image/label.

    Examples
    --------
    >>> TRAIN_DATA_PATH = '/path/to/train/folder'
    >>> x_train, y_train = read_imgs(TRAIN_DATA_PATH, [0, 1])
    """
    tmp = []
    labels = []
    for cls in classes:
        print(cls)
        fnames = glob.glob(os.path.join(dataset_path, str(cls), '*.png'))
        logging.info(f'Read images from class {cls}')
        tmp += [imread(fname) for fname in fnames]
        labels += len(fnames)*[cls]

    return np.array(tmp, dtype='float32')[..., np.newaxis]/255, np.array(labels)

def get_data(train_path = train_path ,test_path = test_path, validation_split=split,
                img_height=60, img_width=60):
    """acquires data from designated folder.
    Returns
    -------
    train, val, test: tf.Dataset
        keras.Dataset type"""
    train = image_dataset_from_directory(
    train_path,
    validation_split=split,
    subset="training",
    seed=123, color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=1)

    val = image_dataset_from_directory(
    train_path,
    validation_split=split,
    subset="validation",
    seed=123,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=1)

    test = image_dataset_from_directory(
    test_path,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=1)
    return train, val, test

callbacks = [EarlyStopping(monitor='val_accuracy', min_delta=5e-3, patience=20, verbose=1),
                ReduceLROnPlateau(monitor='val_accuracy', factor=0.25, min_delta=1e-4,patience=10, verbose=1)]

def plot(history):
    """Plot loss and accuracy
    .....
    Parameters
    ----------
    history: keras History obj
        model.fit() return
   """

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc)+1)
    #Train and validation accuracy 
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    #Train and validation loss 
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show(block=False)