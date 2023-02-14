import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codice")))
from classes import Data, Model
from utils import delete_directory



class DataTests(unittest.TestCase):
    def test(self):
        d_aug_0 = Data(augmented = 0)
        self.assertEqual(len(d_aug_0.X), len(d_aug_0.y))
        X, y = d_aug_0.get_random_images(22)
        self.assertEqual(len(X), len(y), 22)
        #X is not settable
        d_aug_0.X = X
        self.assertFalse(np.array_equal(d_aug_0.X, X))
        

        d_aug_1 = Data(augmented = 1)
        self.assertEqual(len(d_aug_1.X), len(d_aug_1.y))
        X, y = d_aug_1.get_random_images(4)
        self.assertEqual(len(X), len(y), 4)
        d_aug_1.X = X
        self.assertFalse(np.array_equal(d_aug_1.X, X))
        delete_directory(d_aug_1.path) 
   
        '''d_wave_0 = Data(augmented = 0, wavelet = 1)
        self.assertEqual(len(d_wave_0.X), len(d_wave_0.y))
        X, y = d_wave_0.get_random_images(24)
        self.assertEqual(len(X), len(y), 24)
        d_wave_0.X = X
        self.assertFalse(np.array_equal(d_wave_0.X, X))
        delete_directory(d_wave_0.path)

        d_wave_1 = Data(augmented = 1, wavelet = 1)
        self.assertEqual(len(d_wave_1.X), len(d_wave_1.y))
        delete_directory('augmented_data')
        delete_directory(d_wave_1.path)'''
    
class ModelTests(unittest.TestCase):
    def test(self):
        model = self.init()
        self.get_predictions(model=model)

    def init(self):
        data = Data()
        model = Model(data)
        self.assertEqual(len(model.X), len(model.y), len(data))
        X, y = data.get_random_images(22)
        #X, y non settable
        model.X = X
        model.y = y
        self.assertFalse(np.array_equal(model.X, X))
        self.assertFalse(np.array_equal(model.y, y))
        with self.assertRaises(FileNotFoundError):
            model.selected_model = 'fake_dir'
        return model
    
    def get_predictions(self, model):
        X = np.linspace(0., 5., 13)
        with self.assertRaises(FileNotFoundError):
            model.get_predictions(X, models_list=['fake_model'])
        with self.assertRaises(TypeError):
            model.get_predictions(X, models_list=58)

        models_list = [os.path.join('models', 'model_0')]
        if os.path.isdir(os.path.join('models', 'model_0')):
            with self.assertRaises(TypeError):
                model.get_predictions('test', models_list=models_list)
            
            # check if initial values in y (42.) are replaced with predictions
            y = model.get_predictions(X, models_list)
            self.assertFalse(42. in y)










if __name__ == '__main__':
    os.chdir('..')
    unittest.main()
