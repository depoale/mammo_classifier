"""kfold code: really don't get how test accuracy is this shitty since it is basically 
the same code used in models.py (where test accuracy is always above 0.93) """

 
from utils import wave_set, str2bool, grad_plot
from models import set_hyperp, get_search_spaze_size
import numpy as np
import argparse
import os
from classes import Data, Model
import torch
 
 
if __name__=='__main__':
    os.chdir('..')
    parser = argparse.ArgumentParser(
        description="Mammography classifier"
    )
    
    parser.add_argument(
        "-aug",
        "--augmented",
        metavar="",
        type=str2bool,
        help="Whether to perform data augmentation procedure",
        default=False,
    )
    
    parser.add_argument(
        "-wave",
        "--wavelet",
        metavar="",
        type=str2bool,
        help="Whether to apply wavelet filter",
        default=False,
    )

    parser.add_argument(
        "-wave_family",
        "--wavelet_family",
        metavar="",
        type=str,
        help="Which wavelet family (between 'sym3' and 'haar') has to be used to realize the filter",
        default=['sym3'],

    )


    parser.add_argument(
        "-threshold",
        "--threshold",
        metavar="",
        type=float,
        help="threshold of wavelet coefficients in terms of the standard deviation of their distributions (do not go over 2!)",
        default=[1.5],

    )
    
    parser.add_argument(
        "-or",
        "--overwrite",
        metavar="",
        type=str2bool,
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
        default=[1,2,3,4],
    )
    
    
    parser.add_argument(
        "-conv_in",
        "--Conv2d_init",
        metavar="",
        nargs='+',
        type=int,
        help="List of values for the hypermodel's conv2d initial value",
        default=[5, 10, 15, 20],
    )
    
    parser.add_argument(
        "-dropout",
        "--dropout_rate",
        metavar="",
        nargs='+',
        type=float,
        help="List of values for the hypermodel's dropout rate",
        default=[0.1, 0.05, 0.0],
    )

    parser.add_argument(
        "-sf",
        "--searching_fraction",
        metavar="",
        type=float,
        help="Fraction of the hyperparamiters space explored during hypermodel search",
        default=0.25,
    )

    parser.add_argument(
        "-gcam",
        "--gradcam",
        metavar="",
        type=int,
        help="Number of random images to visualize using gradCAM for each class",
        default=3,
    )

    args = parser.parse_args()

    # -----Training process---------------------

    #1. initialize dataset using user picked values
    wave_settings = wave_set(args)
    data = Data(augmented=args.augmented, wavelet=args.wavelet, wave_settings=wave_settings)

    #2. set chosen hyperparameters and get number of trials
    set_hyperp(args)
    space_size = get_search_spaze_size()
    max_trials = np.rint(args.searching_fraction*space_size)

    #3. create and train the model
    model = Model(data=data, overwrite=args.overwrite, max_trials=max_trials)
    model.train()

    #4. visualize test data with gradCAM
    '''test_data = Data()
    test_data.path='total_data'
    X_test, y_test = test_data.get_random_images(size= 10)  #solo per provare, da cambiare ASSOLUTAMENTE
    print(X_test.shape)
    ensemble = torch.load('trained_ensemble')
    ensemble.eval()
    with torch.inference_mode():
        X_test = torch.from_numpy(X_test.astype('float32'))
        outputs = ensemble(X_test)
        print(outputs)
        print(y_test)'''
