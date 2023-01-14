"""first trial kinda naive, all hps set randomly by yours truly. 
no hps tuner, no cv, used hold out to determine test accuracy
nevertheless test accuracy 0.88 ca :)"""

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

""" 
used to convert dataset to 'png' estension (supported by keras.utils.image_dataset_from_directory)   
for file in os.listdir('datasets/Mammography_micro/Test/0'):
    filename, extension  = os.path.splitext(file)
    if extension == ".pgm":
        new_file = "{}.png".format(filename)
        with Image.open(os.path.join('datasets/Mammography_micro/Test/0',file)) as im:
            im.save(new_file)
 """

batch_size = 72
img_height = 60
img_width = 60
split = 0.3

train_path = 'data_png/Train'
test_path = 'data_png/Test'

def get_data(train_path = 'data_png/Train',test_path = 'data_png/Test',validation_split=0.3,
                img_height=60, img_width=60):
    """acquires data from designated folder.
    Returns
    -------
    train, val, test: BatchData ????
        keras.Dataset type"""
    train = image_dataset_from_directory(
    train_path,
    validation_split=split,
    subset="training",
    seed=123, color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size)

    val = image_dataset_from_directory(
    train_path,
    validation_split=split,
    subset="validation",
    seed=123,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size)

    test = image_dataset_from_directory(
    test_path,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size)
    return train, val, test

#model
def get_model():
    """creates model: first layer rescales the grayscale values from [0,255] to [0,1]"""
    model = Sequential()
    model.add(Input(shape=(img_height,img_width,1)))
    #model.add(Rescaling(scale=1./255.))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(Adam(learning_rate=1e-3), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()
    return model

def get_bigger_model():
    """creates model: first layer rescales the grayscale values from [0,255] to [0,1]"""
    model = Sequential()
    model.add(Input(shape=(img_height,img_width,1)))
    #model.add(Rescaling(scale=1./255.))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(Adam(learning_rate=1e-3), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()
    return model

def cnn_model(shape=(60, 60, 1), verbose=False):
    """
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
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1', input_shape=shape))
    model.add(MaxPooling2D((2, 2), name='maxpool_1'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
    model.add(MaxPooling2D((2, 2), name='maxpool_2'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
    model.add(MaxPooling2D((2, 2), name='maxpool_3'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
    model.add(MaxPooling2D((2, 2), name='maxpool_4'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu', name='dense_2'))
    model.add(Dense(128, activation='relu', name='dense_3'))
    model.add(Dense(1, activation='sigmoid', name='output'))

    model.compile(loss='binary_crossentropy', optimizer= Adam(learning_rate=1e-3), metrics=['accuracy'])
    
    if verbose:
      model.summary()
  
    return model



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
    plt.show()

if __name__ == '__main__':
    model = cnn_model()
    train, val, test = get_data()
    history = model.fit(train, batch_size=batch_size , epochs=1000, validation_data=val, callbacks=callbacks)

    plot(history=history)
    print(f'test accuracy: {round(model.evaluate(test)[1],3)}')