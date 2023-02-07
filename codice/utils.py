import os
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import glob
import logging
from skimage.io import imread
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sn
import pandas as pd
import shutil
from PIL import Image
import argparse


split=0.35

def read_imgs(dataset_path, classes):
    """Function reading all the images in a given folder which already contains
    two subfolder.

    Parameters
    ----------
        dataset_path : str
            Path to the image folder.
        classes : list
            0 and 1 mean normal tissue and microcalcification clusters, respectively.

    Returns
    -------
        array: numpy_array
            Array containing the value of image/label.

    Examples
    --------
    >>> TRAIN_DATA_PATH = '/path/to/train/folder'
    >>> x_train, y_train = read_imgs(TRAIN_DATA_PATH, [0, 1])
    """
    tmp = []
    labels = []
    for cls in classes:
        fnames = glob.glob(os.path.join(dataset_path, str(cls), '*.png'))
        logging.info(f'Read images from class {cls}')
        tmp += [imread(fname) for fname in fnames]
        labels += len(fnames)*[cls]


    return np.array(tmp, dtype='float32')[..., np.newaxis]/255, np.array(labels)

callbacks = [EarlyStopping(monitor='val_accuracy', min_delta=5e-3, patience=20, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=1e-4,patience=10, verbose=1)]

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

def wave_set(args):
    wave_settings = {
            'wavelet_family' : args.wavelet_family,
            'threshold': args.threshold

    }
    return wave_settings

    
def delete_directory(directory_path):
    shutil.rmtree(directory_path)

def comparison_plot(names, dimension, mean):
    plt.figure('Comparison plot')
    fig, ax = plt.subplots(figsize=(10,10))

    plt.xlabel('Effective free parameters')
    plt.ylabel('MEE')

    #scatter plot
    for i, txt in enumerate(names):
        ax.errorbar(dimension[i], mean[i], label=names[i], fmt='.')
        ax.annotate(txt, (dimension[i], mean[i]))
    plt.savefig(os.path.join('images', 'comparison.pdf'))
    #plt.legend()
    plt.show()

def create_new_dir(new_dir_path):
    os.makedirs(new_dir_path)

def save_image(saving_directory_path, image_matrix):
    plt.imsave(saving_directory_path, image_matrix, cmap='gray', format='png')

def convert_to_grayscale(image_path):
    Image.open(image_path).convert('L').save(image_path)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
                








