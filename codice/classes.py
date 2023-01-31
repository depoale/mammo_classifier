from utils import read_imgs

class Data():
    def __init__(self, augmented, wavelet):
        """class used to choose and initialize dataset
        ...
        Attributes
        ----------
        augmentes: Bool
            whether to perform data augmentation
        wavelet: Bool
            whether to use wavelet procedure """
        
        self._PATH = 'total_data'
        
        
        if wavelet:
            # create wavelet directory and set _PATH to that directory
            pass
        if augmented:
            # augment data found in _PATH and set _PATH to that directory
            pass
        self.set_data(self._PATH)
    
    def set_data(self, directory):
        self.X, self.y = read_imgs(directory, [0,1])
