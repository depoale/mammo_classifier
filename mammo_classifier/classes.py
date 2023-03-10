"""Custom-made classes"""
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt
import shutil
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from keras.models import load_model
import torch
import statistics as stats

import shutup
shutup.please()
import warnings 
warnings.filterwarnings('ignore')

from utils import read_imgs, callbacks, create_new_dir, save_image, convert_to_grayscale
from plots import ROC, get_confusion_matrix, plot, plot_mean_stdev
from hypermodel import hyp_tuning_model
from tools_for_Pytorch import WeightInitializer, WeightNormalizer, pytorch_linear_model
from ensemble import train_ensemble

img_width = 60
img_height = 60

wave_settings_default = {
    'wavelet_family': 'sym3',
    'threshold': 1.5
}


class Data:
    """class used to choose and initialize dataset
        ...
        Attributes
        ----------
        augmented: bool
            whether to perform data augmentation
        wavelet: bool
            whether to use wavelet filtering 
        wave_settings: dict
            settings for wavelet filtering
        path: str
            path to dataset directory
            
        Methods
        -------
        set_data(directory) -> X, y
            returns arrays of images and lables from directory
        shuffle_data -> X, y
            returns shuffled data
        aug
            performes data augmentation
        wave
            performs wavelet filtering
        get_random_images(size, classes=None) -> X, y
            returns random images from one or both classes
        
    """
     
    def __init__(self, augmented=True, wavelet=False, wave_settings=wave_settings_default, path='dataset'):
       
        #default path to dataset
        self._path = path
        
        # augmentation procedure
        if augmented:
            self.aug()
        
        #wavelet filtering
        if wavelet:
            self.wave(wave_settings)
            
        
        # get images from directory
        self._X, self._y = self.set_data(self._path)
        self._X, self._y = self.shuffle_data()
        self.len = len(self._X)
    
    @property
    def X(self):
        return self._X
    
    #X not settable
    @X.setter
    def X(self, X_new):
        pass

    @property
    def y(self):
        return self._y
    
    #y not settable
    @y.setter
    def y(self, y_new):
        pass

    @property
    def path(self):
        return self._path
    
    #path not settable
    @path.setter
    def path(self, directory):
        pass

    def __len__(self):
        return self.len

    def __getitem__(self, index):    
        return self._X[index], self._y[index] 
    
    def set_data(self, directory):
        X, y = read_imgs(directory, [0,1])
        return X, y
    
    def shuffle_data(self):
        assert len(self._X) == len(self._y)
        p = np.random.permutation(len(self._X))
        return self._X[p], self._y[p]

    def aug(self):
        """Augmentation procedure"""
        # setting new dataset directory to store augmented images
        IMGS_DIR ='augmented_data'

        for cl in ['0', '1']:
            #create a sub-dir for each class
            create_new_dir(os.path.join(f'{IMGS_DIR}', cl))
            dataset_path = os.path.join(f'{self._path}', cl)
            names = os.listdir(dataset_path)
            #copy each image of the original dataset into the new direcrory
            for name in names:
                shutil.copy(os.path.join(f'{self._path}', cl, f'{name}'), os.path.join(f'{IMGS_DIR}', cl))

        #set augmentation transformations
        datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=3,
        height_shift_range=3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        validation_split=0)
        
        #create new images class-wise
        for one_class in os.listdir(self._path):
            dir_path = os.path.join(f'{IMGS_DIR}', one_class)
            gen = datagen.flow_from_directory(
                self._path,
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
        
        #set augmented dataset path
        self._path = IMGS_DIR

    
    def wave(self, wave_settings):

        """Performs a Wavelet-based filtering on the input images (images from the
        original dataset or augmented images) and then saves the new images in 'wavelet_data'
        directory"""
        
        #importing and calling MATLAB as a computational engine
        import matlab.engine
        eng = matlab.engine.start_matlab()
        
        #user-selected parameters

        #setting the Wavelet family to be used 
        wave = wave_settings['wavelet_family']
        #decomposition level
        N = 3 
        #setting the threshold for decomposistion coefficients in terms of their stdev
        Q =  wave_settings['threshold'] 
   
        IMGS_DIR = 'wavelet_data'

        for cl in ['0', '1']:
            #creating a new directory for wavelet images
            create_new_dir(os.path.join(f'{IMGS_DIR}', cl))
            #setting the path to images which will be processed
            dataset_path = os.path.join(f'{self._path}', cl)
            names = os.listdir(dataset_path)
            
            for name in names:
                #reading images from the choosen dataset
                I = eng.imread(os.path.join(dataset_path, f'{name}'))
                #converting MATLAB matrices into numpy arrays
                I = np.asarray(I)
                
                #performing the 2-D Wavelet decomposition
                c, s = eng.wavedec2(I, N, wave, nargout=2)
                
                """s is the bookkeping matrix: it contains the dimensions of the wavelet
                coefficients by level and is used to parse the wavelet decomposition vector c"""

                """c is the vector of decomposition coefficients: it first contains the unrolled 
                approximation matrix, and then the three detail matrices (Horizontal, Vertical, 
                Diagonal) for each of the 3 decomposition levels """

                #converting MATLAB matrices into numpy arrays
                s = np.asarray(s)
                c = np.asarray(c)
                #dimensional adjustment
                c = c[0]

                #extraction of the first level approximation and detail coefficients
                A1 = eng.appcoef2(c,s,wave,1, nargout=1)
                H1,V1,D1 = eng.detcoef2('all',c,s,1, nargout=3)

                #extraction of the second level approximation and detail coefficients
                A2 = eng.appcoef2(c,s,wave,2, nargout=1)
                H2,V2,D2 = eng.detcoef2('all',c,s,2, nargout=3)

                #extraction of the third level approximation and detail coefficients
                A3 = eng.appcoef2(c,s,wave,3, nargout=1)
                H3,V3,D3 = eng.detcoef2('all',c,s,3, nargout=3)

                """Approximation and detail coefficients are stored in the c vector.
                It cointains the unrolled coefficient matrices as it follows:
                [A3(:); H3(:); V3(:); D3(:); H2(:); V2(:); D2(:); H1(:); V1(:); D1(:)]."""

                """We will create a vector of labels where:
                First level detail coefficients (H1,V1,D1) are labelled as 1
                Second level detail coefficients (H2,V2,D2) are labelled as 2
                Third level detail coefficients (H3,V3,D3) are labelled as 3
                Third level approximation coefficients (A3) are labelled as 0"""

                #setting the size of labels' vector
                size_CM = eng.prod(s,2, nargout=1)
                #converting MATLAB matrices into numpy arrays
                size_CM = np.asarray(size_CM)

                #initializing labels' vector
                c_labels= eng.zeros(1,size_CM[0])
                #converting MATLAB matrices into numpy arrays
                c_labels = np.asarray(c_labels)
                #dimensional adjustement
                c_labels = c_labels[0]

                for il in range(1, N+1):
                    #creating a vector of "1" to set labels' vector
                    ones = eng.ones(1,3*np.double(size_CM[il]))
                    #converting MATLAB matrices into numpy arrays
                    ones = np.asarray(ones)
                    #dimensional adjustement
                    ones = ones[0]
                    #setting labels' vector
                    c_labels = np.concatenate((c_labels, np.double(N+1-il)*ones))
                

                #setting the stdev of coefficients' ditribution at level 1
                std1=np.double(eng.std(c[c_labels==1], nargout=1))
                #setting the stdev of coefficients' ditribution at level 2
                std2=np.double(eng.std(c[c_labels==2], nargout=1))
                #setting the stdev of coefficients' ditribution at level 3
                std3=np.double(eng.std(c[c_labels==3], nargout=1))

                """We can set thresholds of coefficients in terms of the stdev of their
                distributions and set to zero the 'low spatial frequency approximation' 
                information while keeping only the 'high spatial frequency details' that
                exceed a certain number Q of standard deviations""" 

                c_mod = c.copy()
                c_mod.setflags(write=1)
                c_mod[c_labels==0]=0

                c_mod[(c_labels==1)&(abs(c)<Q*std1)]=0
                c_mod[(c_labels==2)&(abs(c)<Q*std2)]=0
                c_mod[(c_labels==3)&(abs(c)<Q*std3)]=0

                #reconstructing filtered images
                I_rec = eng.waverec2(c_mod,s,wave, nargout=1)
                #converting MATLAB matrices into numpy arrays
                I_rec = np.asarray(I_rec)

                #saving wavelet_filtered images in the new directory
                save_image(os.path.join(f'{IMGS_DIR}', cl, f'{name}.png'), I_rec)
                #converting saved images to grayscale
                convert_to_grayscale(os.path.join(f'{IMGS_DIR}', cl, f'{name}.png'))

        
        #setting the self path to the new wavelet-filtered images dataset
        self._path = IMGS_DIR

    def get_random_images(self, size:int, classes=None):
        """Extract random elements from the dataset
        .....
        Parameters
        ----------
        size: int
            number of random images 
        classes: list
            list of classes from which we want to extract random images
            default: None means that images will be extracted from all of the classes"""
        
        #extract images from the user-given list
        if classes is not None:
            idx = []
            for cl in classes:
                # create a list patterns belonging to classes list 
                idx.append(np.where(self._y == cl)[0])
            
            idx = np.squeeze(np.array(idx))
            # extract random indices from that list
            rand_idx = np.random.randint(0, len(idx), size=size)
            rand_idx = idx[rand_idx]
        else:
            # random indices regardless of the class the patter belongs to
            rand_idx = np.random.randint(0, self.len, size=size)

        # setting the random indices
        X = self._X[rand_idx]
        y = self._y[rand_idx]
        return X, y

class Model:
    """Perform hyperparameters search, k-fold and train ensemble
    ...
    Attributes
    ----------
    X: np.array
    y: np.array
    max_trials: int
        maximum number of trials in tuner search
    overwrite: bool
        overwrite implies making hps search (not fast)
    k: int
        k-fold parameter. Default 5
    modelBuilder: keras.Hypermodel
        hypermodel
    models_list:list
        list of path to pretrained model (ensemble feeder)
    selected_model: str
        path to a single trained model. Default is ensemble winner
    
    Methods
    -------
    train
        performs hyperparameters search, k-fold and trains ensemble
    tuner(X_dev, y_dev, modelBuilder, i, k=5)->best_hps, best_model
        performs hyperparameters search for the current fold (i)
    fold(k=5)
        performs kfold & hps search in each fold
    get_predictions(self, X=None, models_list=None) -> y
        returns an array of model predictions
    get_ensemble
        create and train ensemble starting from previously obtained models
    """
    def __init__(self, data, fast=True, max_trials=10, k=5):
        self._X = data.X
        self._y = data.y
        self.max_trials = max_trials
        self.overwrite = not fast  #overwrite implies making hps search (not fast)
        self.k = k   # k-fold parameter
        self.modelBuilder = hyp_tuning_model  #hypermodel
        self.models_list = []  # list of path to pretrained models (ensemble feeder)
        self._selected_model = ''    #path to a selected model (default: ensemble winner)

    @property
    def X(self):
        return self._X
    
    #X not settable
    @X.setter
    def X(self, X_new):
        pass

    @property
    def y(self):
        return self._y
    
    #y not settable
    @y.setter
    def y(self, y_new):
        pass

    @property
    def selected_model(self):
        return self._selected_model
    
    @selected_model.setter
    def selected_model(self, model_name):
        if not os.path.isdir(model_name):
            raise FileNotFoundError(f'No such file or directory {model_name}')
        self._selected_model = model_name


    def train(self):
        """Performs hyperparameters search, k-fold and trains ensemble"""
        self.fold()
        self.get_ensemble()

    def tuner(self, X_dev, y_dev, modelBuilder, i, k=5):
        """ Performs hyperparameters search for the current fold (i) using keras-tuner if overwrite is true, 
        else takes default hps.
        Then builds the model, returns the best hyperparameters and the model.
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
        
        Returns
        -------
        best_hps: dict
            Dictionary with hps name and selected values
        best_model: keras.Model
            Model built using those hps
        """
        # set path for tuner
        tuner_dir = os.path.join('tuner', f'tuner_{i}')
        if self.overwrite :
            project_name = 'tuner'
        else:
            # pre-made search
            project_name = 'base'
        
        # tuner settings and search
        # score: val_accuracy so that model with good generalization capability are rewarded 
        tuner = kt.BayesianOptimization(modelBuilder, objective='val_accuracy', max_trials=self.max_trials, 
                                        overwrite=self.overwrite, directory=tuner_dir, project_name=project_name)
        tuner.search(X_dev, y_dev, epochs=50, validation_split=1/(k-1), batch_size=60, 
                    callbacks=callbacks, verbose=1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models()[0]
        return best_hps, best_model

    def fold(self, k=5):
        """Performs kfold & hps search in each fold. Shows some plots to assess the performance of each fold.
        ...
        Parameters
        ----------
        k: int
            k-fold parameter. Default 5"""
        #initialize lists to keep track of each fold's performance
        test_acc = []
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

        # here model selection (hold-out) and model assessment (k-fold) are peformed
        fold  = KFold(n_splits=self.k, shuffle=True, random_state=42)
        for i, (dev_idx, test_idx) in enumerate(fold.split(self._X, self._y)):
            #divide development and test sets
            X_dev, X_test = self._X[dev_idx], self._X[test_idx]
            y_dev, y_test = self._y[dev_idx], self._y[test_idx]
            print('--------------')
            print(f'FOLD {i+1}')
            print('--------------')
            #tuner search with self.tuner
            best_hps, best_model = self.tuner(X_dev, y_dev, self.modelBuilder, i, k=5)
            best_hps_list.append(best_hps)

            #train best model and assess its performance onto test set
            history = best_model.fit(X_dev, y_dev, epochs=100, batch_size=60, validation_split=1/(k-1), callbacks=callbacks)
            accuracy= round(best_model.evaluate(X_test, y_test)[1],3)

            # append this fold's accuracy on test set
            test_acc.append(accuracy)
            
            #add this fold's results to the plots
            plot(history=history, i=i)
            ROC(X_test, y_test=y_test, model=best_model, color=colors[i], i=i+1, mean_fpr=mean_fpr, tprs=tprs, aucs=aucs)
            get_confusion_matrix(X_test, y_test=y_test, model=best_model, i=i+1)

            #save the model and add its path to self.models_list 
            model_path = os.path.join('models', f'model_{i}')
            best_model.save(model_path)
            self.models_list.append(model_path) 

        # plot mean and stdev in ROC curve plot
        plot_mean_stdev(tprs, mean_fpr, aucs)
    
        # print stats and chosen hyperparametrs
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
        plt.show()
    

    def get_predictions(self, X=None, models_list=None):
        """Creates and returns an array of model predictions. Each column corrispond to one expert's 
        predictions (ensemble input). When X=None the predictions are evaluated on self._X, otherwise on whatever 
        X is passed.
        ...
        Parameters
        ----------
        X: np.array
            Array to predict. Default None means that self._X is predicted
        models_list: list
            list of paths to pre-trained models (experts) that will give predictions 
        
        Returns
        -------
        y: np.array
            Array of predictions"""
        
        #check and set X
        if X is None:
            X = self._X
        elif not isinstance(X, np.ndarray):
            raise TypeError(f'Expected np.ndarray got {type(X)}')
        
        #check and set models list
        if models_list is None:
            models_list = self.models_list
        elif isinstance(models_list, list):
            for element in models_list:
                if not os.path.isdir(element):
                    raise FileNotFoundError(f'No such file or directory {element}')
        else: raise TypeError(f'Expected list got {type(models_list)}')
                
        
        #initialise predictions array
        y = np.full(shape=(len(X), len(self.models_list)), fill_value=42.)

        # add each expert predictions column-wise
        for count, expert in enumerate(self.models_list):
            expert = load_model(expert)
            y[:, count] = np.squeeze(expert.predict(X))
        return y

    def get_ensemble(self):
        """Create and train ensemble starting from previously obtained models"""
        #get each expert's predictions
        X = self.get_predictions()

        # transform to tensors
        X = torch.from_numpy(X.astype('float32'))
        y = torch.from_numpy(self._y.astype('float32'))

        # split train and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=24)
        
        # create dataset for external test (data stored in 'External_test')
        test_data = Data(path ='External_test')

        # get random images from this dataset and get each expert's predictions
        X_test, y_test = test_data.get_random_images(size=25, classes=[1])
        X_test = self.get_predictions(X_test)

        # transform to tensors
        X_test = torch.from_numpy(X_test.astype('float32'))
        y_test = torch.from_numpy(y_test.astype('float32'))

        # linear model performs weighted average 
        model = pytorch_linear_model(in_features=len(self.models_list), out_features=1)   
    
        """random weights initialization (and normalisation)
        the weights represent the "reliability" of a model among the commitee, so they are bound 
        in range (0.01, 1) and their sum must add up to 1"""
        initializer=WeightInitializer()
        model.apply(initializer)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.9999))

        # applied after each optimiser step to clip and normalise the updated weights
        normalizer = WeightNormalizer()

        #ensemble training
        weights, final_acc, test_acc = train_ensemble(model, optimizer, normalizer, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=20)
        #Ensemble stats
        print(f'Final accuracy: {final_acc}')
        print(f'Test accuracy: {test_acc}')

        # set selected_model to be the most reliable in the commitee
        weights = torch.tensor(weights.data).numpy()
        best_idx = np.where(weights == weights.max())[0][0]
        self._selected_model = os.path.join('models', f'model_{best_idx}')

        plt.show()


