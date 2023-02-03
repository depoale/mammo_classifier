import unittest
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codice")))
from classes import Data




class DataTests(unittest.TestCase):
    def test_inputs(self):
        d_aug_0 = Data(augmented = 0)
        self.assertEqual(len(d_aug_0.X), len(d_aug_0.y))

        d_aug_1 = Data(augmented = 1)
        self.assertEqual(len(d_aug_1.X), len(d_aug_1.y))

        wave_settings_default = {
            'wavelet_family' : 'sym3',
            'decomposition_level': 3,
            'threshold': 1
        }

        d_wave_0 = Data(augmented = 0, wavelet = 1, wave_settings = wave_settings_default)
        self.assertEqual(len(d_wave_0.X), len(d_wave_0.y))

        d_wave_1 = Data(augmented = 1, wavelet = 1, wave_settings = wave_settings_default)
        self.assertEqual(len(d_wave_1.X), len(d_wave_1.y))

        


        
 


if __name__ == '__main__':
    os.chdir('..')

    unittest.main()
