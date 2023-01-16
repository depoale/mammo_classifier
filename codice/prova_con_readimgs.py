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
import random
from utils import get_data, plot, callbacks
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets"))
) 
from keras.utils import image_dataset_from_directory
from utils import read_imgs
from models import cnn_classifier

PATH='total_data'

""" TRAIN_PATH = os.path.join(PATH, 'Train')
TEST_PATH = os.path.join(PATH, 'Test') """

 
X, y = read_imgs(PATH, [0, 1])
c = list(zip(X, y))

random.shuffle(c)
model = cnn_classifier()
X_train, X_val, X_test= X[:573], X[573:673], X[673:]
y_train, y_val, y_test= y[:573], y[573:673], y[673:]
history = model.fit(X_train,y_train, batch_size=1 , epochs=50, validation_data=(X_val, y_val), callbacks=callbacks)

plot(history=history)
plt.show()
print(f'test accuracy: {round(model.evaluate(X_test, y_test)[1],3)}') 