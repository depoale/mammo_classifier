"""kfold code: really don't get how test accuracy is this shitty since it is basically 
the same code used in models.py (where test accuracy is always above 0.93) """

 
from utils import  set_hps, wave_set
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
from models import callbacks
from matplotlib import pyplot as plt
import argparse
from classes import Data, Model
 
 
if __name__=='__main__':
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
        type=list,
        help="Which wavelet family (between 'sym3' and 'haar') has to be used to realize the filter",
        default=['sym3'],

    )

    parser.add_argument(
        "-dec_level",
        "--decomposition_level",
        metavar="",
        type=list,
        help="Decomposition level of the wavelet analysis",
        default=[3],

    )

    parser.add_argument(
        "-threshold",
        "--threshold",
        metavar="",
        type=list,
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
        type=list,
        help="List of values for the hypermodel's depth",
        default=[1,2,3],
    )
    
    parser.add_argument(
        "-units",
        "--Dense_units",
        metavar="",
        type=list,
        help="List of values for the hypermodel's number of hidden units",
        default=[256],
    )
    
    parser.add_argument(
        "-conv_in",
        "--Conv2d_init",
        metavar="",
        type=list,
        help="List of values for the hypermodel's conv2d initial value",
        default=[10, 20, 30],
    )
    
    parser.add_argument(
        "-dropout",
        "--dropout_rate",
        metavar="",
        type=list,
        help="List of values for the hypermodel's dropout rate",
        default=[0.0, 0.05],
    )
    
    parser.add_argument(
        "-kernel_size",
        "--kernel_size",
        metavar="",
        type=list,
        help="List of values for the hypermodel's kernel_size",
        default=[3, 5],
    )

    args = parser.parse_args()

    ###############
    #1. initialize dataset using user picked values
    wave_settings = wave_set(args)
    Data(augmented=args.augmented, wavelet=args.wavelet, wave_settings=wave_settings)
    hps = set_hps(args)
    #Model(Data=Data, )


