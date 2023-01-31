from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Sequential
import numpy as np
from utils import read_imgs

shape = (60,60,1)

def model1():
    model = Sequential()
    model.add(Conv2D(16, (3,3), padding='same', input_shape=shape))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    return model

def model2():
    model = Sequential()
    model.add(Conv2D(16, (3,3), padding='same', input_shape=shape))
    model.add(Flatten())
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    return model

def model3():
    model = Sequential()
    model.add(Conv2D(16, (3,3), padding='same', input_shape=shape))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    return model

X_train, y_train = read_imgs('data_png/Train', [0,1])
models_list = [model1, model2, model3]

if __name__ =='__main__':
    for model in models_list:
        model=model()
        model.fit(X_train, y_train, epochs=30)
        model.save(f'{model}')
