from utils import read_imgs, set_hps, callbacks, ROC, get_confusion_matrix, plot
from models import hyp_tuning_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
#import matlab.engine
from sklearn.model_selection import KFold
import keras_tuner as kt
from PIL import Image
import statistics as stats
from sklearn.metrics import auc
from ensemble import train_ensemble
import keras

img_width = 60
img_height = 60

class Data:
    def __init__(self, augmented=False, wavelet=False, wave_settings={}):
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
            #self.wave()
            pass
        
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

    
    """ def wave(self, wave_settings):
        eng = matlab.engine.start_matlab()
        
        wave = wave_settings['wavelet_family'] 
        N = wave_settings['decomposition_level'] 
        Q =  wave_settings['threshold'] 
        
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
 """
class Model:
    def __init__(self, Data, hps: dict, overwrite:bool):
        self.X = Data.X
        self.y = Data.y
        self.hps = self.set_hps(hps)
        self.overwrite = overwrite
        self.modelBuilder = hyp_tuning_model
        self.models_list = []

    def train(self):
        self.fold()

    def tuner(self, X_dev, y_dev, modelBuilder, i, k=5):
        tuner = kt.BayesianOptimization(modelBuilder, objective='accuracy', max_trials=5, overwrite=self.overwrite, directory=f'tuner_{i}')
        tuner.search(X_dev, y_dev, epochs=20, validation_split=1/(k-1), batch_size=32, 
                    callbacks=callbacks, verbose=1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models()[0]
        return best_hps, best_model
    
    def retrain_and_save(self, X, y, best_hps_list, modelBuilder, i):
        for i, hps in enumerate(best_hps_list):
            model = modelBuilder(hps)
            model.fit(X, y, epochs=100,validation_split=0.2, callbacks=callbacks)
            model.save(f'model_{i}')  
            self.models_list.append(f'model_{i}') 

    def plot_mean_stdev(self,tprs, mean_fpr, aucs ):
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

    def fold(self, k=5):
        """Performs kfold & hps search in each fold.
        Returs best hps list (one entry for each fold)"""
        #initialize lists to keep track of each fold's performance
        test_acc=[]
        best_hps_list=[]
        
        #preparation for figures
        plt.figure('ROC - Testing')
        plt.title('ROC - Testing')
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure('Confusion Matrices')
        colors = ['green', 'red', 'blue', 'darkorange', 'gold']

        # here model selection and model assessment are peformed (k-fold + hold-out)
        fold  = KFold(n_splits=k, shuffle=True, random_state=42)
        for i, (dev_idx, test_idx) in enumerate(fold.split(self.X, self.y)):
            #divide development and test sets
            X_dev, X_test = self.X[dev_idx], self.X[test_idx]
            y_dev, y_test = self.y[dev_idx], self.y[test_idx]
            print('###############')
            print(f'FOLD {i+1}')
            best_hps, best_model = self.tuner(X_dev, y_dev, self.modelBuilder, i, k=5)
            best_hps_list.append(best_hps)

            #train best model and assess model's performance
            history = best_model.fit(X_dev, y_dev, epochs=100, validation_split=1/(k-1), callbacks=callbacks)
            accuracy= round(best_model.evaluate(X_test, y_test)[1],3)
            test_acc.append(accuracy)
            
            #add this fold's results to the plots
            plot(history=history, i=i)
            ROC(X_test, y_test=y_test, model=best_model, color=colors[i], i=i+1, mean_fpr=mean_fpr, tprs=tprs, aucs=aucs)
            get_confusion_matrix(X_test, y_test=y_test, model=best_model, i=i+1)

        # plot mean and stdev in ROC curve plot
        self.plot_mean_stdev(tprs, mean_fpr, aucs)
    
        print(f'Test Accuracy {test_acc}')
        print(f'Expected accuracy: {stats.mean(test_acc):.2f}+/- {stats.stdev(test_acc):.2f}')
        print(f'best hps:{best_hps_list}')
        plt.show()
        self.retrain_and_save(self.X, self.y, best_hps_list=best_hps_list, modelBuilder=self.modelBuilder, i=i)
        
    def ensemble(self):
        #load previously saved models
        model_0=keras.models.load_model('model_0')
        model_1=keras.models.load_model('model_1')
        model_2=keras.models.load_model('model_2')
        model_3=keras.models.load_model('model_3')
        model_4=keras.models.load_model('model_4')
        X = get_predictions(X, models_list)

        # transform to tensors
        X = torch.from_numpy(X. astype('float32'))
        y = torch.from_numpy(y.astype('float32'))

        X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=VAL_SPLIT, shuffle=True, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=VAL_SPLIT, shuffle=True, random_state=24)

        # model = weighted average     
        model = nn.Sequential(nn.Linear(in_features=len(models_list), out_features=1))

        # weights initialization 
        w_init = weights_init_uniform_fan_in
        model.apply(w_init)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        weights, final_acc = train_ensemble(model, optimizer, X_train, y_train, X_val, y_val, X_test, y_test, name='ensemble')
        print(f'Final accuracy: {final_acc}')
        for w in weights:
            print(w)
        plt.show()


