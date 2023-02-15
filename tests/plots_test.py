import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codice")))
from plots import plot, ROC, plot_mean_stdev, get_confusion_matrix, gCAM_show

if __name__ == '__main__':
    os.chdir('..')
    unittest.main()