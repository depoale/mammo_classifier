from utils import read_imgs
import os
from keras.preprocessing.image import ImageDataGenerator

img_width = 60
img_height = 60

class Data():
    def __init__(self, augmented: bool, wavelet: bool):
        """class used to choose and initialize dataset
        ...
        Attributes
        ----------
        augmentes: bool
            whether to perform data augmentation
        wavelet: bool
            whether to use wavelet procedure """
        
        self._PATH = 'total_data'
        
        
        if wavelet:
            # create wavelet directory and set _PATH to that directory
            pass
        if augmented:
            # augment data found in _PATH and set _PATH to that directory
            self.aug()
            
        self.set_data(self._PATH)
    
    def set_data(self, directory):
        self.X, self.y = read_imgs(directory, [0,1])

    def aug(self):
        IMGS_DIR ='augmented_data'

        datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=3,
        height_shift_range=3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect', #  nearest?
        validation_split=0)
        
        for i, one_class in enumerate(os.listdir(IMGS_DIR)):
            dir_path = os.path.join(os.getcwd(),'augmented_data', one_class)
            gen = datagen.flow_from_directory(
                IMGS_DIR,
                target_size = (img_width, img_height),
                batch_size = 1,
                color_mode = 'grayscale',
                class_mode = None,
                classes = [one_class],
                save_to_dir = dir_path
            )
            #generate & save the images
            for k in range(len(gen)):
                gen.next()

        self._PATH = IMGS_DIR
