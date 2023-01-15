from sklearn.model_selection import KFold
from utils import callbacks
import numpy as np
from matplotlib import pyplot as plt
import statistics as stats


def fold(X, y, model, k=5):
    """performs kfold model assessment & plots accuracy learning curve.
    Parameters:
    ----------
    xxxxxxxxxxdata: keras.preprocessing.image.DirectoryIterator
        result from data augmentation procedure
    X: numpy.array
        data
    y: numpy.array
        labels
    model: tf.keras.Model
        model choosen for training
    k: int
        number of folds. default set to 5 """
    outer_kfold = KFold(n_splits=k, shuffle=False, random_state=None)
    
    acc=[]
    val_acc=[]   
    for dev_idx, test_idx in outer_kfold.split(X, y):
        X_dev, X_test = X[dev_idx], X[test_idx]
        y_dev, y_test = y[dev_idx], y[test_idx]   
        print(len(X_dev), len(X_test))    
        history = model.fit(X_dev,y_dev,validation_split=1/(k-1), epochs=500,verbose=1, batch_size=64, callbacks = callbacks)
        plt.plot(history.history["val_accuracy"], label='validation set')
        plt.plot(history.history["accuracy"], label='training set')
        plt.legend(loc='best')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        #plt.show()

        #model.evaluate returns a tuple containing loss in position 0 & accuracy in pos 1
        acc.append(model.evaluate(X_test, y_test)[1]) 
        val_acc.append(history.history["val_accuracy"][-1])
        if val_acc and history.history["val_accuracy"][-1] == min(val_acc):
            model.save_weights("weights.h5")

    print(acc)
    print(f"Expected acc:{round(stats.mean(acc),2)}+/-{round(stats.variance(acc),2)}")