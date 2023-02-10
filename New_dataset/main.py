import keras
import glob
import logging
import os
from skimage.io import imread
import numpy as np

def read_imgs(dataset_path, classes):
    tmp = []
    labels = []
    for cls in classes:
        fnames = glob.glob(os.path.join(dataset_path, str(cls), '*.png'))
        logging.info(f'Read images from class {cls}')
        tmp += [imread(fname) for fname in fnames]
        labels += len(fnames)*[cls]


    return np.array(tmp, dtype='float32')[..., np.newaxis]/255, np.array(labels)


X, y = read_imgs('NEW_DATA', [0,1])

model = keras.models.load_model('model2')
print(model.evaluate(X,y))