import unittest
import sys
import numpy as np
from matplotlib import pyplot as plt
from cmepda_prj.code import prepro_ale 

class PreproTests(unittest.TestCase):
    def test_sets():
        x_train, y_train = ale.read_imgs(ale.train_dataset_path, [0, 1])
        if len(x_train)!= len(y_train):
            raise ValueError('dimension mismatch') 


if __name__ == '__main__':
    unittest.main()