import unittest
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codice")))
import utils

class UtilsTests(unittest.TestCase):
    def test_read_imgs(self):
        X, y = utils.read_imgs('total_data', [0, 1])
        self.assertEqual(len(X)==len(y))

    def test_str2bool(self):
        self.assertEqual(utils.str2bool('YES')== True)
        self.assertEqual(utils.str2bool('y')== True)
        self.assertEqual(utils.str2bool('0')== False)
        self.assertEqual(utils.str2bool('F')== False)
    
    def test_nearest_square(self):
        self.assertEqual(utils.nearest_square(16)== 4)
        self.assertEqual(utils.nearest_square(12)== 3)
        self.assertEqual(utils.nearest_square(6)== 2)
    
    def test_get_rows_columns(self):
        self.assertEqual(utils.get_rows_columns(16)== (4,4))
        self.assertEqual(utils.get_rows_columns(17)== (4,5))
        self.assertEqual(utils.get_rows_columns(12)== (3,4))
        
        





    
if __name__ == '__main__':
    os.chdir('..')
    unittest.main()

