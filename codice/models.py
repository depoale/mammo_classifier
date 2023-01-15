"""few models. first ones quite naive.
cnn_model has great potential. with this partition btw train and test
got test accuracy = 0.967 :))"""

import os
import sys
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Rescaling, Input
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, BinaryAccuracy
from PIL import Image
from utils import get_data, plot, callbacks

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

batch_size = 64
img_height = 60
img_width = 60
split = 0.3

train_path = 'data_png/Train'
test_path = 'data_png/Test'



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

def cnn_model(shape=(60, 60, 1), learning_rate=1e-3, verbose=False):
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

    model.compile(loss='binary_crossentropy', optimizer= Adam(learning_rate=learning_rate), metrics=['accuracy'])
    
    if verbose:
      model.summary()
  
    return model


if __name__ == '__main__':
    model = cnn_model()
    train, val, test = get_data(train_path='new_data/Train', test_path='new_data/Test')
    history = model.fit(train, batch_size=batch_size , epochs=1000, validation_data=val, callbacks=callbacks)

    plot(history=history)
    plt.show()
    print(f'test accuracy: {round(model.evaluate(test)[1],3)}') 