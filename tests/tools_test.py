import unittest
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codice")))
from classes import Data, Model
from utils import delete_directory
import tools_for_Pytorch as tools



class ToolsTests(unittest.TestCase):
    def test(self):
        model = self.linear()
        self.weights(model=model)


    def linear(self):
        model = tools.pytorch_linear_model(4, 2)
        self.assertEqual(torch.numel(model[0].weight.data), 8)

        model = tools.pytorch_linear_model()
        self.assertEqual(torch.numel(model[0].weight.data), 5)

        return model
    
    def weights(self, model):
        old_weights = model[0].weight.data
        initializer = tools.WeightInitializer()
        model.apply(initializer)
        new_weights = model[0].weight.data
        self.assertFalse(torch.equal(old_weights, new_weights))

        normalizer = tools.WeightNormalizer()
        model[0].weight.data.fill_(1)
        model.apply(normalizer)
        #check if any element is still 1 
        self.assertFalse(1 in model[0].weight.data)
        #check if weights sum is one
        self.assertAlmostEqual(torch.sum(model[0].weight.data), 1)


        






if __name__ == '__main__':
    os.chdir('..')
    unittest.main()
