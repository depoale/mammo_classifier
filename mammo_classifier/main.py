"""Main"""
import numpy as np
import argparse
import os
import time
import keras
import shutup
import warnings 
from utils import wave_dict, hyperp_dict, str2bool, rate, delete_directory
from hypermodel import get_search_space_size
from classes import Data, Model
from gcam import get_gcam_images
from plots import gCAM_show
warnings.filterwarnings('ignore')
shutup.please()
if __name__=='__main__':
    start = time.time()
    os.chdir('..')
    parser = argparse.ArgumentParser(
        description="Mammograms classifier"
    )
    
    parser.add_argument(
        "-aug",
        "--augmented",
        metavar="",
        type=str2bool,
        help="Whether to perform data augmentation procedure. Default: True",
        default=True,
    )
    
    parser.add_argument(
        "-wave",
        "--wavelet",
        metavar="",
        type=str2bool,
        help="Whether to apply wavelet filter. Default: False",
        default=False,
    )

    parser.add_argument(
        "-wave_fam",
        "--wavelet_family",
        metavar="",
        type=str,
        help="Wavelet family choice (between 'sym3' and 'haar'). Default: 'sym3'",
        default='sym3',

    )


    parser.add_argument(
        "-thr",
        "--threshold",
        metavar="",
        type=float,
        help="Threshold of wavelet coefficients in terms of the stdev of their distributions (do not go over 2!). Default: 1.5",
        default=1.5,

    )
    
    parser.add_argument(
        "-fast",
        "--fast_execution",
        metavar="",
        type=str2bool,
        help="If True avoid hyperparameters search and use the pre-saved hyperpar. Default: False",
        default=False,
    )
    
    parser.add_argument(
        "-depth",
        "--net_depth",
        metavar="",
        nargs='+',
        type=int,
        help="List of values for the hypermodel's depth",
        #default=[1,2,3,4],
        default=[1,3,5],
    )
    
    
    parser.add_argument(
        "-dim",
        "--Conv2d_dim",
        metavar="",
        nargs='+',
        type=int,
        help="List of values for the hypermodel's conv2d initial value",
        default=[15,20, 25],
    )
    
    parser.add_argument(
        "-dropout",
        "--dropout_rate",
        metavar="",
        nargs='+',
        type=rate,
        help="List of values for the hypermodel's dropout rate",
        default=[0.1, 0.05, 0.0],
    )

    parser.add_argument(
        "-sf",
        "--searching_fraction",
        metavar="",
        type=rate,
        help="Fraction of the hyperparamiters space explored during hypermodel search. Default: 0.25",
        default=0.25,
    )

    parser.add_argument(
        "-gcam",
        "--gradcam",
        metavar="",
        type=int,
        help="Number of random images to visualize using gradCAM. Default: 6",
        default=6,
    )

    args = parser.parse_args()

    # -----Training process---------------------

    #1. initialize dataset using user picked values
    wave_settings = wave_dict(args.wavelet_family, args.threshold)
    data = Data(augmented=args.augmented, wavelet=args.wavelet, wave_settings=wave_settings)

    #2. set chosen hyperparameters and get number of trials
    hyperp_dict=hyperp_dict(args.net_depth, args.Conv2d_dim, args.dropout_rate)
    space_size = get_search_space_size()
    max_trials = np.rint(args.searching_fraction*space_size)

    #3. create and train the model
    model = Model(data=data, fast=args.fast_execution, max_trials=max_trials)
    model.train()

    #4. check what the most reliable model has learnt using gradCAM
    best_model = keras.models.load_model(model.selected_model)
    num_images = args.gradcam
    if num_images > 25:
        print('Showing 25 images using gradCAM')
        num_images = 25
    rand_images, _ = data.get_random_images(size=num_images, classes=[1])
    preds = best_model.predict(rand_images)
    get_gcam_images(rand_images, best_model)
    gCAM_show(preds=preds)