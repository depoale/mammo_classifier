import unittest
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codice")))
from utils import read_imgs

class read_imgsTests(unittest.TestCase):
    def test(self):
        X,y = read_imgs('total_data', [7576,87687])
        print(len(X), len(y))






'''class UtilsTests(unittest.TestCase):
    def test_(self):
        x_train, y_train, x_test, y_test= ale.get_data()
        if len(x_train)!= len(y_train) or len(x_test)!= len(y_test):
            raise ValueError('dimension mismatch')''' 
    
if __name__ == '__main__':
    os.chdir('..')
    unittest.main()

