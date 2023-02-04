"""kfold code: really don't get how test accuracy is this shitty since it is basically 
the same code used in models.py (where test accuracy is always above 0.93) """

 
from utils import wave_set
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
from models import callbacks, set_hyperp
from matplotlib import pyplot as plt
import argparse
import os
from classes import Data, Model
 
 
if __name__=='__main__':
    os.chdir('..')
    parser = argparse.ArgumentParser(
        description="Mammography classifier"
    )
    
    parser.add_argument(
        "-aug",
        "--augmented",
        metavar="",
        type=bool,
        help="Whether to perform data augmentation procedure",
        default=False,
    )
    
    parser.add_argument(
        "-wave",
        "--wavelet",
        metavar="",
        type=bool,
        help="Whether to apply wavelet filter",
        default=False,
    )

    parser.add_argument(
        "-wave_family",
        "--wavelet_family",
        metavar="",
        nargs='+',
        type=str,
        help="Which wavelet family (between 'sym3' and 'haar') has to be used to realize the filter",
        default=['sym3'],

    )


    parser.add_argument(
        "-threshold",
        "--threshold",
        metavar="",
        nargs='+',
        type=float,
        help="threshold of wavelet coefficients in terms of the standard deviation of their distributions (do not go over 2!)",
        default=[1.5],

    )
    
    parser.add_argument(
        "-or",
        "--overwrite",
        metavar="",
        type=bool,
        help="Whether to perform hyperparameters search or use the previously saved hyperpar",
        default=False,
    )
    
    parser.add_argument(
        "-depth",
        "--net_depth",
        metavar="",
        nargs='+',
        type=int,
        help="List of values for the hypermodel's depth",
        default=[1,2,3],
    )
    
    
    parser.add_argument(
        "-conv_in",
        "--Conv2d_init",
        metavar="",
        nargs='+',
        type=int,
        help="List of values for the hypermodel's conv2d initial value",
        default=[10, 20, 30],
    )
    
    parser.add_argument(
        "-dropout",
        "--dropout_rate",
        metavar="",
        nargs='+',
        type=float,
        help="List of values for the hypermodel's dropout rate",
        default=[0.0, 0.05],
    )

    args = parser.parse_args()

    ###############
    #1. initialize dataset using user picked values
    wave_settings = wave_set(args)
    data = Data(augmented=args.augmented, wavelet=args.wavelet, wave_settings=wave_settings)
    #2. set chosen hyperparameters 
    set_hyperp(args)
    #3. create and train the model
    model = Model(data=data)
    model.train()


