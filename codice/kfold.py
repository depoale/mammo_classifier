import os

from matplotlib import pyplot as plt
import keras_tuner as kt

from sklearn.model_selection import KFold
import random
import statistics as stats
from sklearn.utils import shuffle
from utils import  plot, callbacks
import seaborn as sn
import pandas as pd

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from utils import read_imgs, ROC, get_confusion_matrix
from models import cnn_classifier, hyp_tuning_model, make_model
from sklearn.metrics import roc_curve, auc, confusion_matrix

PATH='augmented_data'


def fold(X, y, k, modelBuilder):
    """didn't add implementation for ensemble, so do not use"""
    test_acc=[]
    fold  = KFold(n_splits=k, shuffle=True, random_state=42)
    colors = ['green', 'red', 'blue', 'darkorange', 'gold']
    for i, (dev_idx, test_idx) in enumerate(fold.split(X, y)):
        X_dev, X_test = X[dev_idx], X[test_idx]
        y_dev, y_test = y[dev_idx], y[test_idx]
        model = modelBuilder()
        history = model.fit(X_dev, y_dev, epochs=100, validation_split=0.2, batch_size=64,callbacks=callbacks)
        accuracy= round(model.evaluate(X_test, y_test)[1],3)
        plot(history=history, i=i)
        print(f'test accuracy: {accuracy}')
        test_acc.append(accuracy)
        print('#########')
        print(f'Sani: train {len(y_dev[y_dev==0])}, test {len(y_test[y_test==0])}')
        print(f'Malati: train {len(y_dev[y_dev==1])}, test {len(y_test[y_test==1])}')
        ROC(X_test, y_test=y_test, model=model, color=colors[i-1], i=i)
        get_confusion_matrix(X_test, y_test=y_test, model=model, i=i)


    print(test_acc)
    print(f'Expected accuracy: {round(stats.mean(test_acc),3)}+/- {round(stats.stdev(test_acc),3)}')
    plt.show()

def retrain_and_save(X, y, best_hps_list, modelBuilder):
    for i, hps in enumerate(best_hps_list):
        model = modelBuilder(hps)
        model.fit(X, y, epochs=100,validation_split=0.2, callbacks=callbacks)
        model.save(f'model_{i}')

def fold_tuner(X, y, k, modelBuilder, overwrite=True):
    """Performs kfold & hps search in each fold.
        Returs best hps list (one entry for each fold)"""
    test_acc=[]
    best_hps_list=[]
    colors = ['green', 'red', 'blue', 'darkorange', 'gold']

    fold  = KFold(n_splits=k, shuffle=True, random_state=42)

    for i, (dev_idx, test_idx) in enumerate(fold.split(X, y)):
        #divide development and test sets
        X_dev, X_test = X[dev_idx], X[test_idx]
        y_dev, y_test = y[dev_idx], y[test_idx]
        #set tuner
        print('###############')
        print(f'FOLD {i}')
        tuner = kt.BayesianOptimization(modelBuilder, objective='accuracy', max_trials=5, overwrite=overwrite, directory=f'tuner_{i}')
        tuner.search(X_dev, y_dev, epochs=20, validation_split=1/(k-1), batch_size=32, 
                    callbacks=callbacks, verbose=1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_hps_list.append(best_hps)
    
        best_model = tuner.get_best_models()[0]
        history = best_model.fit(X_dev, y_dev, epochs=100, validation_split=1/(k-1), callbacks=callbacks)
        accuracy= round(best_model.evaluate(X_test, y_test)[1],3)
        test_acc.append(accuracy)
        
        plot(history=history, i=i)
        ROC(X_test, y_test=y_test, model=best_model, color=colors[i], i=i)
        get_confusion_matrix(X_test, y_test=y_test, model=best_model, i=i)

    print(test_acc)
    print(f'Expected accuracy: {round(stats.mean(test_acc),3)}+/- {round(stats.stdev(test_acc),3)}')
    print(f'best hps:{best_hps_list}')
    plt.show()
    retrain_and_save(X, y, best_hps_list=best_hps_list, modelBuilder=modelBuilder)


if __name__ == '__main__':
    print(os.getcwd())
    X, y = read_imgs(PATH, [0, 1])
    X, y = shuffle(X, y)
    fold_tuner(X, y, k=5, modelBuilder=hyp_tuning_model, overwrite=True)
    #plt.show()
    