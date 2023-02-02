import unittest
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codice")))
from classes import Data


class DataTests(unittest.TestCase):
    def test_inputs(self):
        d1=Data(augmented=1)
        self.assertEqual(len(d1.X), len(d1.y))
 


if __name__ == '__main__':

    unittest.main()
