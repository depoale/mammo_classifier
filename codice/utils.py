import os
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import glob
import logging
from skimage.io import imread
import numpy as np
import shutil
from random import shuffle
from PIL import Image
import argparse
from hypermodel import set_hyperp

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
        if not os.path.isdir(os.path.join(dataset_path, str(cls))):
            raise FileNotFoundError(f'No such file or directory {os.path.join(dataset_path, str(cls))}') 
        fnames = glob.glob(os.path.join(dataset_path, str(cls), '*.png'))
        logging.info(f'Read images from class {cls}')
        tmp += [imread(fname) for fname in fnames]
        labels += len(fnames)*[cls]


    return np.array(tmp, dtype='float32')[..., np.newaxis]/255, np.array(labels)

callbacks = [EarlyStopping(monitor='val_accuracy', min_delta=5e-3, patience=20, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=1e-4, patience=10, verbose=1)]

def wave_dict(wavelet_family, threshold):
    """Creates dictionary containing wavelet settings"""
    wave_dict = {
            'wavelet_family' : wavelet_family,
            'threshold': threshold

    }
    return wave_dict

def hyperp_dict(net_depth, Conv2d_init, dropout_rate):
    """Creates dictionary containing wavelet settings"""
    hyperp_dict = {
            'depth' : list(set(net_depth)),
            'Conv2d_init': list(set(Conv2d_init)),
            'dropout' : list(set(dropout_rate))

    }
    set_hyperp(hyperp_dict)
    return hyperp_dict

    
def delete_directory(directory_path):
    shutil.rmtree(directory_path)

def create_new_dir(new_dir_path):
    if os.path.isdir(new_dir_path):
        delete_directory(new_dir_path)
    os.makedirs(new_dir_path)

def save_image(saving_directory_path, image_matrix):
    plt.imsave(saving_directory_path, image_matrix, cmap='gray', format='png')

def convert_to_grayscale(image_path):
    Image.open(image_path).convert('L').save(image_path)

def str2bool(v):
    """String to bool conversion"""
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        raise argparse.ArgumentTypeError('Boolean value expected.')
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_rate(v):
    if not 0. <= v <= 1.:
                raise ValueError(f'Rate value should be between 0 and 1 not {v}')

def rate(values):
    """Check type and value of rate quantities"""
    if isinstance(values, list):
        for v in values:
            if isinstance(v, float):
                check_rate(v)
            else: 
                raise argparse.ArgumentTypeError(f'Argument invalid: expected float got {type(v)}')
    elif isinstance(values, float):
        check_rate(values)
    else:
        raise argparse.ArgumentTypeError(f'Argument invalid: expected float got {type(values)}')

    return values

# functions for gCAM plot
def nearest_square(limit:int):
    """Returns the sqrt of the highest square less or equal than limit"""
    if not isinstance(limit, int):
        raise TypeError(f'Expected type int got {type(limit)}')
    answer = 0
    while (answer+1)**2 <= limit:
        answer += 1
    return answer
    
def get_rows_columns(size:int):
    """Create a matrix starting from the nearest square"""
    if not isinstance(size, int):
        raise TypeError(f'Expected type int got {type(size)}')
    n = nearest_square(size)
    rows = n
    columns = n
    while rows*columns < size:
        columns += 1
    return rows, columns



                








