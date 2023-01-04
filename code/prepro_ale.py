from skimage.io import imread, imread_collection
import skimage
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from PIL import Image


"""show one of the images"""
im=Image.open('datasets/Mammography_micro/Train/0/0002s1_2_0.pgm_2.pgm', mode='r', formats=None)
im.show()

def read_imgs(dataset_path, classes):
    """A function to unpack images and labels from a folder
    ...
    Parameters
    ----------
    dataset_path: str
        path to the dataset

    classes: list
        list of classes according to their folders
        eg: the first class will be dataset_path/classes[0]
    """
    tmp = []
    labels = []
    for cls in classes:
     # Returns all the images filenames contained in a certain folder
        fnames = glob.glob(os.path.join(dataset_path, str(cls), '*.pgm'))
     # Read, with a list comprehension, all the images listed before
        tmp += [ imread(fname) for fname in fnames ]
    # Populate the labels list with the label of the read image
        labels += len(fnames)*[cls]
    return np.array(tmp, dtype='float32')[..., np.newaxis]/255, np.array(labels)


train_dataset_path = 'datasets/Mammography_micro/Train'
x_train, y_train = read_imgs(train_dataset_path, [0, 1])

test_dataset_path = 'datasets/Mammography_micro/Test'
x_test, y_test = read_imgs(test_dataset_path, [0, 1])

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

'''x_train is the array of the images, in our case the first dimension is the number of
read images, the second and the third are the dimensions of each image, and the last one
 represents the color channel: (sample_dimension, image_width, image_height, color_channel). 
 While y_train is the array containing the label of each image (0 or 1).'''

