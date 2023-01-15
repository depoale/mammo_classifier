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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets"))
) 
from keras.utils import image_dataset_from_directory

split=0.3

def get_data(train_path = 'data_png/Train',test_path = 'data_png/Test', validation_split=split,
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

callbacks = EarlyStopping(monitor='val_accuracy', min_delta=5e-3, patience=20, verbose=1)
                #ReduceLROnPlateau(monitor='val_accuracy', factor=0.25, patience=10, verbose=1))

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