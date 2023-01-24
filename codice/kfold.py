import os

from matplotlib import pyplot as plt
import keras_tuner as kt

from sklearn.model_selection import KFold
import random
import statistics as stats
from sklearn.utils import shuffle
from utils import get_data, plot, callbacks


from utils import read_imgs
from models import cnn_classifier, hyp_tuning_model

PATH='total_data'

def fold(X, y, k, modelBuilder):
    test_acc=[]
    fold  = KFold(n_splits=k, shuffle=True, random_state=42)
    for dev_idx, test_idx in fold.split(X, y):
        X_dev, X_test = X[dev_idx], X[test_idx]
        y_dev, y_test = y[dev_idx], y[test_idx]
        model = modelBuilder()
        history = model.fit(X_dev, y_dev, epochs=100, validation_split=1/(k-1), callbacks=callbacks)
        accuracy= round(model.evaluate(X_test, y_test)[1],3)
        plot(history=history)
        print(f'test accuracy: {accuracy}')
        test_acc.append(accuracy)

    print(test_acc)
    print(f'Expected accuracy: {round(stats.mean(test_acc),3)}+/- {round(stats.stdev(test_acc),3)}')
    plt.show()

def fold_tuner(X, y, k, modelBuilder):
    test_acc=[]
    best_hps_list=[]
    fold  = KFold(n_splits=k, shuffle=True, random_state=42)
    for dev_idx, test_idx in fold.split(X, y):
        X_dev, X_test = X[dev_idx], X[test_idx]
        y_dev, y_test = y[dev_idx], y[test_idx]
        tuner = kt.BayesianOptimization(modelBuilder, objective='accuracy', max_trials=5, overwrite=True, directory='tuner')
        tuner.search(X_dev, y_dev, epochs=50, validation_split=1/(k-1), batch_size=32, 
                    callbacks=callbacks, verbose=1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_hps_list.append(best_hps)
        #tuner.results_summary()

        best_model = tuner.get_best_models()[0]
        print(f'best_hps:{best_hps}')
        history = best_model.fit(X_dev, y_dev, epochs=100, validation_split=1/(k-1), callbacks=callbacks)
        accuracy= round(best_model.evaluate(X_test, y_test)[1],3)
        plot(history=history)
        print(f'test accuracy: {accuracy}')
        test_acc.append(accuracy)

    print(test_acc)
    print(f'Expected accuracy: {round(stats.mean(test_acc),3)}+/- {round(stats.stdev(test_acc),3)}')
    print(f'best hps:{best_hps_list}')
    plt.show()

def fold_tuner_auto(X, y, k, modelBuilder):
    test_loss_list=[]
    best_hps_list=[]
    fold  = KFold(n_splits=k, shuffle=True, random_state=42)
    for dev_idx, test_idx in fold.split(X, y):
        X_dev, X_test = X[dev_idx], X[test_idx]
        y_dev, y_test = y[dev_idx], y[test_idx]
        tuner = kt.BayesianOptimization(modelBuilder, objective='mse', max_trials=5, overwrite=True, directory='tuner')
        tuner.search(X_dev, y_dev, epochs=50, validation_split=1/(k-1), batch_size=32, 
                    callbacks=callbacks, verbose=1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_hps_list.append(best_hps)
        #tuner.results_summary()

        best_model = tuner.get_best_models()[0]
        print(f'best_hps:{best_hps}')
        history = best_model.fit(X_dev, y_dev, epochs=100, validation_split=1/(k-1), callbacks=callbacks)
        test_loss= round(best_model.evaluate(X_test, y_test),3)
        #plot(history=history)
        print(f'test loss: {test_loss}')
        test_loss_list.append(test_loss)

    print(test_loss_list)
    print(f'Expected accuracy: {round(stats.mean(test_loss_list),3)}+/- {round(stats.stdev(test_loss_list),3)}')
    print(f'best hps:{best_hps_list}')
    plt.show()



if __name__ == '__main__':
    print(os.getcwd())
    X, y = read_imgs(PATH, [0, 1])
    X, y = shuffle(X, y)
    fold(X, y, k=5, modelBuilder=cnn_classifier)
    