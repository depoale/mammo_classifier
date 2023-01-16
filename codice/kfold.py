from models import cnn_model, cnn_classifier 
from utils import get_data, plot
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
from models import callbacks
from matplotlib import pyplot as plt
from keras.utils import image_dataset_from_directory
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets"))
) 
img_height = 60
img_width  =60
split = 0.3
path = 'total_data'


 
if __name__=='__main__':
    data = image_dataset_from_directory(
    path,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=1)
    data.shuffle(42)
 
    inputs = np.concatenate(list(data.map(lambda x, y:x)))
    targets = np.concatenate(list(data.map(lambda x, y:y)))
    print(inputs.shape, targets.shape)

    kfold = KFold(n_splits=8, shuffle=True, random_state=42)
    test_acc=[]

    for train, test in kfold.split(inputs, targets):
        print(train.shape, test.shape)
        model = cnn_classifier()
        history = model.fit(inputs[train], targets[train],  batch_size=1 , 
                            epochs=200, validation_split=1/7, callbacks=callbacks)
        accuracy= round(model.evaluate(inputs[test], targets[test],)[1],3)
        plot(history=history)
        print(f'test accuracy: {accuracy}')
        test_acc.append(accuracy)

    print(test_acc)
    plt.show()
