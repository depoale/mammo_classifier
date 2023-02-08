from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
import seaborn as sn

def plot(history, i):
    """Plot loss and accuracy
    .....
    Parameters
    ----------
    history: keras History obj
        model.fit() return
   """

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc)+1)
    #Train and validation accuracy 
    plt.figure(f'Fold {i}',figsize=(8, 8))
    plt.title(f'Learning curve and accuracy for fold{i}')
    plt.subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='darkorange')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    #Train and validation loss 
    plt.subplot(1, 2, 2)
    #plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs_range, loss, label='Training Loss', color='blue')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='darkorange')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    #plt.show(block=False)

def ROC(x_test, y_test, model, color, i, mean_fpr, tprs, aucs):
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_accuracy}')
    
    preds_test = model.predict(x_test, verbose = 1)
    fpr, tpr, thresholds = roc_curve(y_test, preds_test)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    #print(f'y_test: {y_test}, preds_test: {preds_test}, thresholds: {thresholds}')
    #print(f'len fpr: {len(fpr)} and len tpr: {len(tpr)} for fold {i}')
    print(f'len x_test: {len(x_test)} and len y_test: {len(y_test)}')
    #print(f'fpr: {fpr}, tpr: {tpr}')
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    print(f'AUC = {roc_auc}')
    
    plt.figure('ROC - Testing')
    #plt.title('ROC - Testing')
    lw = 2
    plt.plot(fpr, tpr, color = color, alpha=0.45, lw = lw, label = f'{i} ROC curve (area = {round(roc_auc, 2)})')
    plt.plot([0, 1], [0, 1], color = 'purple', lw = lw, linestyle = '--')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    

def get_confusion_matrix(x_test, y_test, model, i):
    preds_test = np.rint(model.predict(x_test, verbose = 1))
    my_confusion_matrix = confusion_matrix(y_test, preds_test)
    print(f'confusion matrix of fold {i}: {my_confusion_matrix}')
    df_cm = pd.DataFrame(my_confusion_matrix, index = ['Negative', 'Positive'],
                  columns = ['Negative', 'Positive'])
    plt.figure('Confusion Matrices')
    plt.subplot(2,3,i)
    plt.title(f'Fold {i}', fontsize=13, loc='left')
    sn.heatmap(df_cm, annot=True)
    plt.xlabel('Predicted label', fontsize=7)
    plt.ylabel('Actual label', fontsize=7)

def comparison_plot(names, dimension, mean):
    fig, ax = plt.subplots(figsize=(7,7))
    plt.title('Comparison plot')
    plt.xlabel('Effective free parameters')
    plt.ylabel('MSE')

    #scatter plot
    print('shapes_ dentro',dimension, mean, names )
    for i, txt in enumerate(names):
       
        ax.errorbar(dimension[i], mean[i], label=names[i], fmt='.')
        ax.annotate(txt, (dimension[i], mean[i]))
    plt.savefig(os.path.join('images', 'comparison.pdf'))
    #plt.legend()
    plt.show(block=False)

def grad_plot(data, size:int):
    rnd_idx = np.random.randint(0, 700, size = size)
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 2
    images = data.X
    labels = data.y
    for i, idx in enumerate(rnd_idx):
        ax=fig.add_subplot(rows, columns, i)
        ax.title.set_text(f'Label = {labels[idx]}')
        plt.imshow(images[idx], cmap='gray')
    plt.show()