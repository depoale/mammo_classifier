from utils import read_imgs, callbacks, create_new_dir, save_image, convert_to_grayscale, shuffle_data
from plots import  ROC, get_confusion_matrix, plot, comparison_plot, plot_mean_stdev
from models import hyp_tuning_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt
from keras.utils.layer_utils import count_params
from random import shuffle
import tensorflow as tf
import shutil
#import matlab.engine
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from keras.models import load_model
import torch
import torch.nn as nn
import statistics as stats
from sklearn.metrics import auc
from tools_for_Pytorch import weights_init_ones, WeightNormalizer
from ensemble import train_ensemble
from PIL import Image

img_width = 60
img_height = 60

wave_settings_default = {
    'wavelet_family': 'sym3',
    'threshold': 1.5
}


class Data:
    def __init__(self, augmented=False, wavelet=False, wave_settings=wave_settings_default):
        """class used to choose and initialize dataset
        ...
        Attributes
        ----------
        augmentes: bools
            whether to perform data augmentation
        wavelet: bool
            whether to use wavelet procedure 
        wave_settings: dict
            settings for wavelet procedure"""
        
        #default path to dataset
        self._PATH = 'total_data'

        # path to dataset
        @property
        def path(self):
            return self._PATH
        
        @path.setter
        def path(self, directory):
            if type(directory) != str:
                raise TypeError(f'Expected str type got {type(directory)}')
            self._PATH = directory
        
        # augmentation procedure
        if augmented:
            self.aug()
        
        #wavelet procedure
        if wavelet:
            #self.wave(wave_settings)
            pass
        
        self.set_data(self._PATH)
        self.X, self.y = shuffle_data(self.X, self.y)
        self.len = len(self.X)

    def __len__():
        return self.len

    def __getitem__(self, index):    
        return self.X[index], self.y[index] 
    
    def set_data(self, directory):
        self.X, self.y = read_imgs(directory, [0,1])

    def aug(self):
        IMGS_DIR ='augmented_data'
        for cl in ['0', '1']:
            create_new_dir(os.path.join(f'{IMGS_DIR}', cl))
            dataset_path = os.path.join(f'{self._PATH}', cl)
            names = os.listdir(dataset_path)
            for name in names:
                shutil.copy(os.path.join(f'{self._PATH}', cl, f'{name}'), os.path.join(f'{IMGS_DIR}', cl))

        datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=3,
        height_shift_range=3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect', #  nearest?
        validation_split=0)
        
        for i, one_class in enumerate(os.listdir(self._PATH)):
            dir_path = os.path.join(f'{IMGS_DIR}', one_class)
            gen = datagen.flow_from_directory(
                self._PATH,
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

    
    """ def wave(self, wave_settings):
        eng = matlab.engine.start_matlab()
        
        wave = wave_settings['wavelet_family'] 
        N = 3 
        Q =  wave_settings['threshold'] 
        
        IMGS_DIR = 'wavelet_data'

        for cl in ['0', '1']:
            create_new_dir(os.path.join(f'{IMGS_DIR}', cl))
            dataset_path = os.path.join(f'{self._PATH}', cl)
            names = os.listdir(dataset_path)
            for name in names:
                I = eng.imread(os.path.join(dataset_path, f'{name}'))
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

                #plt.imsave(os.path.join(f'{IMGS_DIR}', f'{i}', f'{name}.png'), I_rec, cmap='gray', format='png')
                save_image(os.path.join(f'{IMGS_DIR}', cl, f'{name}.png'), I_rec)
                #Image.open(os.path.join(f'{IMGS_DIR}', f'{i}', f'{name}.png')).convert('L').save(os.path.join(f'{IMGS_DIR}', f'{i}', f'{name}.png'))
                convert_to_grayscale(os.path.join(f'{IMGS_DIR}', cl, f'{name}.png'))


    
        self._PATH = IMGS_DIR """

    def get_random_images(self, size:int):
        """Extract random elements from the dataset
        .....
        Parameters
        ----------
        size: int
            number of random images """
        rand_idx = np.random.randint(0, len(self.X), size=size)
        X = self.X[rand_idx]
        y = self.y[rand_idx]
        return X, y

class Model:
    """Perform hyperparameters search, k-fold and train ensemble"""
    def __init__(self, data, overwrite, max_trials):
        self.X = data.X
        self.y = data.y
        self.max_trials = max_trials
        self.overwrite = overwrite
        self.modelBuilder = hyp_tuning_model
        self.models_list = []

    def train(self):
        """Perform hyperparameters search, k-fold and train ensemble"""
        self.fold()
        self.get_ensemble()

    def tuner(self, X_dev, y_dev, modelBuilder, i, k=5):
        """If overwrite is true performes hps search, else takes default hps.
        ....
        Parameters
        ----------
        X_dev: np.array
            X for development/design set (train+validation)
        y_dev: np.array
            y for development/design set (train+validation)
        modelBuilder: keras.Hypermodel
            funtion returning the hypermodel or model if hps are fixed
        i: int
            k-fold index
        k:int
            number of folds. Default 5
        """
        if self.overwrite :
            project_name = 'tuner'
        else:
            project_name = 'base'
        tuner = kt.BayesianOptimization(modelBuilder, objective='val_accuracy', max_trials=self.max_trials, 
                                        overwrite=self.overwrite, directory=f'tuner_{i}', project_name=project_name)
        tuner.search(X_dev, y_dev, epochs=50, validation_split=1/(k-1), batch_size=64, 
                    callbacks=callbacks, verbose=1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models()[0]
        return best_hps, best_model
    
    def retrain_and_save(self, X, y, hps, modelBuilder, i):
        model = modelBuilder(hps)
        model.fit(X, y, epochs=100, batch_size=64, validation_split=0.25, callbacks=callbacks)
        model.save(f'model_{i}')  
        #self.models_list.append(f'model_{i}') 

    def fold(self, k=5):
        """Performs kfold & hps search in each fold.
        Returs best hps list (one entry for each fold)"""
        #initialize lists to keep track of each fold's performance
        test_acc=[]
        dimension=[]
        best_hps_list=[]

        #preparation for figures
        plt.figure('ROC - Testing')
        plt.title('ROC - Testing')
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure('Confusion Matrices')
        plt.title('Confusion Matrices')
        colors = ['green', 'red', 'blue', 'darkorange', 'gold']

        # here model selection and model assessment are peformed (k-fold + hold-out)
        fold  = KFold(n_splits=k, shuffle=True, random_state=42)
        for i, (dev_idx, test_idx) in enumerate(fold.split(self.X, self.y)):
            #divide development and test sets
            X_dev, X_test = self.X[dev_idx], self.X[test_idx]
            y_dev, y_test = self.y[dev_idx], self.y[test_idx]
            print('--------------')
            print(f'FOLD {i+1}')
            print('--------------')
            best_hps, best_model = self.tuner(X_dev, y_dev, self.modelBuilder, i, k=5)
            best_hps_list.append(best_hps)

            #train best model and assess model's performance
            history = best_model.fit(X_dev, y_dev, epochs=100, batch_size=64, validation_split=1/(k-1), callbacks=callbacks)
            accuracy= round(best_model.evaluate(X_test, y_test)[1],3)

            dimension.append(count_params(best_model.trainable_weights))
            test_acc.append(accuracy)
            self.models_list.append(f'model_{i}')
            
            #add this fold's results to the plots
            plot(history=history, i=i)
            ROC(X_test, y_test=y_test, model=best_model, color=colors[i], i=i+1, mean_fpr=mean_fpr, tprs=tprs, aucs=aucs)
            get_confusion_matrix(X_test, y_test=y_test, model=best_model, i=i+1)

            #retrain on the whole dataset and save best model 
            #self.retrain_and_save(self.X, self.y, hps=best_hps, modelBuilder=self.modelBuilder, i=i)
            best_model.save(f'model_{i}')

        # plot mean and stdev in ROC curve plot
        plot_mean_stdev(tprs, mean_fpr, aucs)
        comparison_plot(names=self.models_list, dimension=dimension, accuracy=test_acc)
    
        print(f'Test Accuracy {test_acc}')
        print(f'Expected accuracy: {stats.mean(test_acc):.2f}+/- {stats.stdev(test_acc):.2f}')
        print(f'best hps:')
        for i, hps in enumerate(best_hps_list):
            print(f'--------------------')
            print(f'Model_{i} chosen hps:')
            print(f'--------------------')
            print(f"Depth: {hps.get('depth')}")
            print(f"Conv_in: {hps.get('Conv2d_init')}")
            print(f"Dropout: {hps.get('dropout')}")
            print(f'--------------------')
        plt.show(block=False)
    

    def get_predictions(self, X=None):
        """Creates and returns an array of model predictions. Each column corrispond to one expert preds.
        Used both in training and in assessing the performance of the model (when X=None the predictions are 
        evaluated on self.X, otherwise on whatever X is passed)"""
        if X is None:
            X = self.X
        y = np.empty(shape=(len(X), len(self.models_list)))
        for count, expert in enumerate(self.models_list):
            expert = load_model(expert)
            y[:, count] = np.squeeze(expert.predict(X))
        return y

    def get_ensemble(self):
        """Create and train ensemble starting from models previously obtained"""
        #get each expert's predictions
        X = self.get_predictions()

        # transform to tensors
        X = torch.from_numpy(X.astype('float32'))
        y = torch.from_numpy(self.y.astype('float32'))

        X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, shuffle=True, random_state=24)

        # model = weighted average    
        model = nn.Sequential(nn.Linear(in_features=len(self.models_list), out_features=1, bias=False)
                              )
    
        # weights initialization and bias set to zero not trainable
        w_init = weights_init_ones
        model.apply(w_init)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.08, betas=(0.9, 0.9999))
        # we want the sum of the weights to be one
        normalizer = WeightNormalizer()

        weights, final_acc = train_ensemble(model, optimizer, normalizer, X_train, y_train, X_val, y_val, X_test, y_test, name='ensemble')
        print(f'Final accuracy: {final_acc}')
        for w in weights:
            print(w)
        plt.show()


