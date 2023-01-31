import numpy as np
import keras
from utils import read_imgs
import tensorflow as tf
from tools_for_Pytorch import get_predictions

def evaluate_ensemble(preds, weights):
    ensemble=0
    for pred, weight in zip(preds, weights):
         ensemble += weight*pred
    return ensemble

def weighted_majority (X_train, y_train, X_test, y_test, expert_list, eta):
    #initialize the weights, and normalize
    weights = np.ones(len (expert_list))
    weights = weights/weights.sum()
    init_cum_acc = 0
    for expert in expert_list:
        for pattern, label in zip (X_train, y_train):
            y_pred = np.rint(expert.predict(pattern))
            if y_pred == label:
                init_cum_acc+=1

    init_ypred = init_ensemble.predict(X_test, y_test)
    init_cum_acc = get_ensemble(init_ypred, weights)
    print(f'iniziale:{init_cum_acc}')
    for expert, weight in zip(expert_list, weights):
        for pattern, label in zip (X_train, y_train):
            y_pred = expert.predict(pattern)
            if y_pred == label:
                pass
            #update weight (diminish weight in the sum if wrong)
            else: weight-=(eta*weight)
    #normalize
    weights = weights/weights.sum()
    final_ensemble = get_ensemble(expert_list, weights)
    final_cum_acc = final_ensemble.evaluate(X_test, y_test)[1]
    print(f'finale: {final_cum_acc}')
    return final_ensemble, weights

if __name__=='__main__':
    X_train, y_train = read_imgs('data_png/Train', [0,1])
    X_test, y_test = read_imgs('data_png/Test', [0,1])
    model1=keras.models.load_model('model1')
    model2=keras.models.load_model('model2')
    model3=keras.models.load_model('model3')
    models_list = [model1, model2, model3]
    y = get_predictions(X_train, models_list)
    print(y.shape)
