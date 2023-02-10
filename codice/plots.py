from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
import seaborn as sn
import glob
from utils import get_rows_columns
from skimage.io import imread

def plot(history, i):
    """Learning curve and accuracy plot.
    .....
    Parameters
    ----------
    history: keras.History 
        returned by model.fit()
    i: int
        k-fold index
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
    """Each fold's contribution to the ROC curve plot
     .....
    Parameters
    ----------
    x_test: np.array
    y_test: np.array
    model: keras.Model
    color: str
    i: int
        k-fold index
    mean_fpr: np.array
    tprs: np.array
    aucs: np.array
    """
    _, test_accuracy = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_accuracy}')
    
    preds_test = model.predict(x_test, verbose = 1)
    fpr, tpr, _ = roc_curve(y_test, preds_test)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    print(f'len x_test: {len(x_test)} and len y_test: {len(y_test)}')
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
    
def plot_mean_stdev(tprs, mean_fpr, aucs):
    """"?????? ?????? ?????? ?????? ?????? ?????? ?????? ?????? ?????? ??????"""
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.figure('ROC - Testing')
    plt.plot(mean_fpr, mean_tpr, color="black", label=f"Mean ROC (AUC = {mean_auc:.2f} $\pm${std_auc:.2f})", lw=2, alpha=0.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.figure('ROC - Testing')
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=f"$\pm$ 1 std. dev.")
    plt.legend(loc='lower right')

def get_confusion_matrix(x_test, y_test, model, i):
    """Each fold's confusion matrix
    .....
    Parameters
    ----------
    x_test: np.array
    y_test: np.array
    model: keras.Model
    i: int
        k-fold's index
    """
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

def comparison_plot(names, dimension, accuracy):
    """MSE-num_weights plot to compare visually each fold's performance in relationship with
        that model's complexity. 
         .....
        Parameters
        ----------
        names: list of str
                models names
        dimension: list of int
                list of models' number of weights
        accuracy: list of float
            list of models' accuracy over test set
        """
    fig, ax = plt.subplots(figsize=(7,7))
    plt.title('Comparison plot')
    plt.xlabel('Effective free parameters')
    plt.ylabel('MSE')

    #scatter plot
    for i, txt in enumerate(names):
       
        ax.errorbar(dimension[i], accuracy[i], label=names[i], fmt='.')
        ax.annotate(txt, (dimension[i], accuracy[i]))
    plt.savefig(os.path.join('images', 'comparison.pdf'))
    #plt.legend()
    plt.show(block=False)

def gCAM_images(preds, cam_path='gCam'):
    """Shows gradCAM images comparing lables and model predictions
     .....
    Parameters
    ----------
    preds: np.array
        array of predictions
    cam_path: str
        path to images
    """
    #get images from cam_path
    tmp = []
    fnames = glob.glob(os.path.join(cam_path, '*.png'))
    tmp += [imread(fname) for fname in fnames]
    images = np.array(tmp, dtype='float32')[...]/255

    #create image
    fig = plt.figure(figsize=(8, 8))
    size = len(images)
    rows, columns = get_rows_columns(size)
    for i in range(size):
        ax = fig.add_subplot(rows, columns, i+1)
        ax.set_axis_off()
        ax.title.set_text(f'Label=1 Pred={preds[i][0]:.1f}')
        plt.imshow(images[i])

    #plt.savefig('gcam.pdf')
    plt.show()