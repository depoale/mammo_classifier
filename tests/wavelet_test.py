import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codice")))
from classes import Data
from utils import delete_directory



class DataTests_wavelet(unittest.TestCase):
    def test(self):
        
        d_wave_0 = Data(augmented = 0, wavelet = 1)
        self.assertEqual(len(d_wave_0.X), len(d_wave_0.y))
        X, y = d_wave_0.get_random_images(24)
        self.assertEqual(len(X), len(y), 24)
        d_wave_0.X = X
        self.assertFalse(np.array_equal(d_wave_0.X, X))
        delete_directory(d_wave_0.path)

        d_wave_1 = Data(augmented = 1, wavelet = 1)
        self.assertEqual(len(d_wave_1.X), len(d_wave_1.y))
        delete_directory('augmented_data')
        delete_directory(d_wave_1.path)
    






if __name__ == '__main__':
    os.chdir('..')
    unittest.main()
