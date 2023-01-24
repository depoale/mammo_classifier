"""data augmentation procedure.
    lowkey shitty bc test(& often val) accuracy fixed around 0.49 :(
    tried tweaking some param in ImageDataGenerator but nothing major happened
    try separating 0 from 1 in generating"""

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from utils import read_imgs, plot, callbacks, split
from models import get_model, get_bigger_model, cnn_model, cnn_classifier
from models import train_path, test_path, img_height, img_width
from kfold import fold, fold_tuner
from sklearn.utils import shuffle
import os
batch_size=64
IMGS_DIR ='total_data'

datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=3,
        height_shift_range=3,
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect', #  nearest?
        validation_split=0)
        
labels= ['0','1']


if __name__=='__main__': 
    print('\n cwd:',os.getcwd())

    for i, one_class in enumerate(os.listdir(IMGS_DIR)):
        print(one_class)
        gen = datagen.flow_from_directory(
            IMGS_DIR,
            target_size = (img_width, img_height),
            batch_size = 1,
            class_mode = None,
            classes = [one_class],
            save_to_dir ='total_classes_'
        )
        for k in range(len(gen)):
            gen.next()

    """ for label in labels:
        path = os.path.join(os.getcwd(),IMGS_DIR, label)
        print(path)
        dir_path = os.path.join(os.getcwd(),'augmented_data', label)
        print(dir_path)
        datagen.flow_from_directory(
                path,
                target_size=(img_width, img_height),
                batch_size=1,
                class_mode = None,
                color_mode='grayscale', 
                save_to_dir='dir_path')
 """