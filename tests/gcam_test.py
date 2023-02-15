import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codice")))
from gcam import make_gradcam_heatmap, get_gcam_images

if __name__ == '__main__':
    os.chdir('..')
    unittest.main()