from utils import create_new_dir, save_image, convert_to_grayscale
import os
import numpy as np
import matlab.engine
import warnings 
warnings.filterwarnings('ignore')

def wave_filtering(self_path, wave_settings):

        """Performs a Wavelet-based filtering procedure on the input images (images from the
        original dataset or augmented images) and then saves the new images in a proper 
        directory"""
        
        #calling MATLAB as a computational engine
        eng = matlab.engine.start_matlab()
        
        #setting the Wavelet family to be used 
        wave = wave_settings['wavelet_family']
        #the decomposition level is already set
        N = 3 
        #setting the threshold for decomposistion coefficients in terms of their stdev
        Q =  wave_settings['threshold'] 
   
        IMGS_DIR = 'wavelet_data'

        for cl in ['0', '1']:
            #creating a new directory for wavelet images
            create_new_dir(os.path.join(f'{IMGS_DIR}', cl))
            #setting the path to images which will be processed
            dataset_path = os.path.join(f'{self_path}', cl)
            names = os.listdir(dataset_path)
            
            for name in names:
                #reading images from the choosen dataset
                I = eng.imread(os.path.join(dataset_path, f'{name}'))
                #converting MATLAB matrices into numpy arrays
                I = np.asarray(I)
                
                #performing the 2-D Wavelet decomposition
                c, s = eng.wavedec2(I, N, wave, nargout=2)
                
                """s is the bookkeping matrix: it contains the dimensions of the wavelet
                coefficients by level and is used to parse the wavelet decomposition vector c"""

                """c is the vector of decomposition coefficients: it first contains the unrolled 
                approximation matrix, and then the three detail matrices (Horizontal, Vertical, 
                Diagonal) for each of the 3 decomposition levels """

                #converting MATLAB matrices into numpy arrays
                s = np.asarray(s)
                c = np.asarray(c)
                #dimensional adjustment
                c = c[0]

                #extraction of the first level approximation and detail coefficients
                A1 = eng.appcoef2(c,s,wave,1, nargout=1)
                H1,V1,D1 = eng.detcoef2('all',c,s,1, nargout=3)

                #extraction of the second level approximation and detail coefficients
                A2 = eng.appcoef2(c,s,wave,2, nargout=1)
                H2,V2,D2 = eng.detcoef2('all',c,s,2, nargout=3)

                #extraction of the third level approximation and detail coefficients
                A3 = eng.appcoef2(c,s,wave,3, nargout=1)
                H3,V3,D3 = eng.detcoef2('all',c,s,3, nargout=3)

                """Approximation and detail coefficients are stored in the c vector.
                It cointains the unrolled coefficient matrices as it follows:
                [A3(:); H3(:); V3(:); D3(:); H2(:); V2(:); D2(:); H1(:); V1(:); D1(:)]."""

                """We will create a vector of labels where:
                First level detail coefficients (H1,V1,D1) are labelled as 1
                Second level detail coefficients (H2,V2,D2) are labelled as 2
                Third level detail coefficients (H3,V3,D3) are labelled as 3
                Third level approximation coefficients (A3) are labelled as 0"""

                #setting the size of labels' vector
                size_CM = eng.prod(s,2, nargout=1)
                #converting MATLAB matrices into numpy arrays
                size_CM = np.asarray(size_CM)

                #initializing labels' vector
                c_labels= eng.zeros(1,size_CM[0])
                #converting MATLAB matrices into numpy arrays
                c_labels = np.asarray(c_labels)
                #dimensional adjustement
                c_labels = c_labels[0]

                for il in range(1, N+1):
                    #creating a vector of "1"
                    ones = eng.ones(1,3*np.double(size_CM[il]))
                    #converting MATLAB matrices into numpy arrays
                    ones = np.asarray(ones)
                    #dimensional adjustement
                    ones = ones[0]
                    #setting labels' vector
                    c_labels = np.concatenate((c_labels, np.double(N+1-il)*ones))
                

                #setting the stdev of coefficients' ditribution at level 1
                std1=np.double(eng.std(c[c_labels==1], nargout=1))
                #setting the stdev of coefficients' ditribution at level 2
                std2=np.double(eng.std(c[c_labels==2], nargout=1))
                #setting the stdev of coefficients' ditribution at level 3
                std3=np.double(eng.std(c[c_labels==3], nargout=1))

                """We can set thresholds of coefficients in terms of the stdev of their
                distributions and set to zero the 'low spatial frequency approximation' 
                information while keeping only the 'high spatial frequency details' that
                exceed a certain number Q of standard deviations""" 

                c_mod = c.copy()
                c_mod.setflags(write=1)
                c_mod[c_labels==0]=0

                c_mod[(c_labels==1)&(abs(c)<Q*std1)]=0
                c_mod[(c_labels==2)&(abs(c)<Q*std2)]=0
                c_mod[(c_labels==3)&(abs(c)<Q*std3)]=0

                #reconstructing filtered images
                I_rec = eng.waverec2(c_mod,s,wave, nargout=1)
                ##converting MATLAB matrices into numpy arrays
                I_rec = np.asarray(I_rec)

                #saving wavelet_filtered images in the new directory
                save_image(os.path.join(f'{IMGS_DIR}', cl, f'{name}.png'), I_rec)
                #converting saved images to grayscale
                convert_to_grayscale(os.path.join(f'{IMGS_DIR}', cl, f'{name}.png'))


