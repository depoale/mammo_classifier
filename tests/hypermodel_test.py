import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codice")))
from hypermodel import set_hyperp, get_search_space_size

class set_args():
    """Fake user-selected arguments"""
    def __init__(self, net_depth, Conv2d_init, dropout_rate):
        self.net_depth = net_depth
        self.Conv2d_init = Conv2d_init
        self.dropout_rate = dropout_rate

class HypermodelTests(unittest.TestCase):
    def test_count_combinations(self):
        args=set_args([1, 1], [1, 2], [1, 2, 3])
        set_hyperp(args)
        self.assertEqual(get_search_space_size(), 6)

        args=set_args([1], [1, 2], [1, 2, 3])
        set_hyperp(args)
        self.assertEqual(get_search_space_size(), 6)

    
    
    


    
if __name__ == '__main__':
    os.chdir('..')
    unittest.main()