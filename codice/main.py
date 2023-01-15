"""kfold code: really don't get how test accuracy is this shitty since it is basically 
the same code used in models.py (where test accuracy is always above 0.93) """

from models import cnn_model 
from utils import get_data, plot
from model_assessment import fold
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
from models import callbacks
from matplotlib import pyplot as plt
 
 
if __name__=='__main__':
    train, val, _ = get_data(train_path='total_data', test_path='data_all/Test')

    train_images = np.concatenate(list(train.map(lambda x, y:x)))
    train_labels = np.concatenate(list(train.map(lambda x, y:y)))

    val_images = np.concatenate(list(val.map(lambda x, y:x)))
    val_labels = np.concatenate(list(val.map(lambda x, y:y)))

    inputs = np.concatenate((train_images, val_images), axis=0)
    targets = np.concatenate((train_labels, val_labels), axis=0)

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    test_acc=[]

    for train, test in kfold.split(inputs, targets):
        model = cnn_model()
        history = model.fit(inputs[train], targets[train],  batch_size=1 , 
                            epochs=200, validation_split=0.25, callbacks=callbacks)
        accuracy= round(model.evaluate(inputs[test], targets[test],)[1],3)
        plot(history=history)
        print(f'test accuracy: {accuracy}')
        test_acc.append(accuracy)

    print(test_acc)
    plt.show()
