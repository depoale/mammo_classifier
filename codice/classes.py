from utils import read_imgs
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
import matlab.engine
from PIL import Image

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
        
        if augmented:
            # augment data found in _PATH and set _PATH to that directory
            self.aug()
        
        if wavelet:
            # create wavelet directory and set _PATH to that directory
            self.wave()
        
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

    
    def wave(self):
        eng = matlab.engine.start_matlab()

        Q = 1 #quante dev std considero nel filtraggio (al massimo due consigliate)
        wave = 'sym3' #quale wavelet utilizzo (anche un altro paio)
        N = 3 #livello di approssimazione (consigliato 3)

        IMGS_DIR = 'wavelet_data'

        os.makedirs(os.path.join(f'{IMGS_DIR}', '0'))
        os.makedirs(os.path.join(f'{IMGS_DIR}', '1'))

        dataset_path_0 = os.path.join(f'{self._PATH}', '0')
        dataset_path_1 = os.path.join(f'{self._PATH}', '1')
        dataset_paths = [dataset_path_0, dataset_path_1]

        names_0 = os.listdir(dataset_path_0)
        names_1 = os.listdir(dataset_path_1)
        names = [names_0, names_1]

        for i in range(0,2,1):
            for name in names[i]:
                I = eng.imread(os.path.join(dataset_paths[i], f'{name}'))
                I = np.asarray(I)
                
                c, s = eng.wavedec2(I, N, wave, nargout=2)
                s = np.asarray(s)
                c = np.asarray(c)
                c = c[0]

                A1 = eng.appcoef2(c,s,wave,1, nargout=1)
                H1,V1,D1 = eng.detcoef2('all',c,s,1, nargout=3)

                A2 = eng.appcoef2(c,s,wave,2, nargout=1)
                H2,V2,D2 = eng.detcoef2('all',c,s,2, nargout=3)

                A3 = eng.appcoef2(c,s,wave,3, nargout=1)
                H3,V3,D3 = eng.detcoef2('all',c,s,3, nargout=3)

                size_CM = eng.prod(s,2, nargout=1)
                size_CM = np.asarray(size_CM)

                c_labels= eng.zeros(1,size_CM[0])
                c_labels = np.asarray(c_labels)
                c_labels = c_labels[0]

                for il in range(1, N+1):
                    ones = eng.ones(1,3*np.double(size_CM[il]))
                    ones = np.asarray(ones)
                    ones = ones[0]
                    c_labels = np.concatenate((c_labels, np.double(N+1-il)*ones))

                std1=np.double(eng.std(c[c_labels==1], nargout=1))
                std2=np.double(eng.std(c[c_labels==2], nargout=1))
                std3=np.double(eng.std(c[c_labels==3], nargout=1))

                c_mod = c.copy()
                c_mod.setflags(write=1)
                c_mod[c_labels==0]=0

                c_mod[(c_labels==1)&(abs(c)<Q*std1)]=0
                c_mod[(c_labels==2)&(abs(c)<Q*std2)]=0
                c_mod[(c_labels==3)&(abs(c)<Q*std3)]=0

                I_rec = eng.waverec2(c_mod,s,wave, nargout=1)
                I_rec = np.asarray(I_rec)

                plt.imsave(os.path.join(f'{IMGS_DIR}', f'{i}', f'{name}.png'), I_rec, cmap='gray', format='png')
                Image.open(os.path.join(f'{IMGS_DIR}', f'{i}', f'{name}.png')).convert('L').save(os.path.join(f'{IMGS_DIR}', f'{i}', f'{name}.png'))

        
        self._PATH = IMGS_DIR

class Model:
    def __init__(self, Data, hps: dict):
        self.X = Data.X
        self.y = Data.y
        self.hps = self.set_hps(hps)

    def set_hps(self, hps):
        pass
