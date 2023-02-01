"""kfold code: really don't get how test accuracy is this shitty since it is basically 
the same code used in models.py (where test accuracy is always above 0.93) """

from models import cnn_model 
from utils import get_data, plot
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
        "-wave",
        "--wavelet",
        metavar="",
        type=bool,
        help="Whether to apply wavelet filter",
        default=False,
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
        "-kerner_size",
        "--kernel_size",
        metavar="",
        type=list,
        help="List of values for the hypermodel's kernel_size",
        default=[3, 5],
    )

    args = parser.parse_args()

    ###############
    #1. initialize dataset using user picked values
    Data(augmented=args.augmented, wavelet=args.wavelet)
    hps = set_hps(args)
    Model(Data=Data, )


