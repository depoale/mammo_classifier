import unittest
import numpy as np
from matplotlib import pyplot as plt
from context import classes 

class DataTests(unittest.TestCase):
    def test_inputs(self):
        classes.Data(augmented=0, wavelet=1)
        classes.Data(augmented=0, wavelet=0)
        classes.Data(augmented=15, wavelet=1)
    
if __name__ == '__main__':
    unittest.main()