import os
import sys
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Rescaling, Input
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
import random
import statistics as stats
from sklearn.utils import shuffle
from utils import get_data, plot, callbacks

from keras.utils import image_dataset_from_directory
from utils import read_imgs
from models import cnn_classifier

PATH='total_data'

""" TRAIN_PATH = os.path.join(PATH, 'Train')
TEST_PATH = os.path.join(PATH, 'Test') """

 
""" X, y = read_imgs(PATH, [0, 1])
model = cnn_classifier()
X, y = shuffle(X, y)
X_train, X_val, X_test= X[:573], X[573:673], X[673:]
y_train, y_val, y_test= y[:573], y[573:673], y[673:]
print(len(X_train), len(X_val), len(X_test))
history = model.fit(X_train,y_train, batch_size=1 , epochs=50, validation_data=(X_val, y_val), callbacks=callbacks)

plot(history=history)
plt.show()
print(f'test accuracy: {round(model.evaluate(X_test, y_test)[1],3)}') 
 """
def fold(X, y, k):
    test_acc=[]
    fold  = KFold(n_splits=k, shuffle=True, random_state=42)
    for dev_idx, test_idx in fold.split(X, y):
        X_dev, X_test = X[dev_idx], X[test_idx]
        y_dev, y_test = y[dev_idx], y[test_idx]
        model = cnn_classifier()
        history = model.fit(X_dev, y_dev, epochs=50, validation_split=1/(k-1), callbacks=callbacks)
        accuracy= round(model.evaluate(X_test, y_test)[1],3)
        plot(history=history)
        print(f'test accuracy: {accuracy}')
        test_acc.append(accuracy)

    print(test_acc)
    print(f'Expected accuracy: {round(stats.mean(test_acc),3)}+/- {round(stats.variance(test_acc),3)}')
    plt.show()


if __name__ == '__main__':
    print(os.getcwd())
    X, y = read_imgs(PATH, [0, 1])
    X, y = shuffle(X, y)
    fold(X, y, k=5)
    