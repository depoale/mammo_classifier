import unittest
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import shutup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codice")))
from classes import Data, Model
from utils import delete_directory

shutup.please()

class DataTests(unittest.TestCase):
    def test_inputs(self):
        d_aug_0 = Data(augmented = 0)
        self.assertEqual(len(d_aug_0.X), len(d_aug_0.y))

        d_aug_1 = Data(augmented = 1)
        self.assertEqual(len(d_aug_1.X), len(d_aug_1.y))
        delete_directory(d_aug_1._PATH)
   
        d_wave_0 = Data(augmented = 0, wavelet = 1)
        self.assertEqual(len(d_wave_0.X), len(d_wave_0.y))
        delete_directory(d_wave_0._PATH)

        d_wave_1 = Data(augmented = 1, wavelet = 1)
        self.assertEqual(len(d_wave_1.X), len(d_wave_1.y))
        delete_directory('augmented_data')
        delete_directory(d_wave_1._PATH)


'''class ModelTests(unittest.TestCase):
    def test_inputs(self):'''


        


        
 


if __name__ == '__main__':
    os.chdir('..')

    unittest.main()
