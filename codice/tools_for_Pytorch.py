import numpy as np
import torch
from torch import nn

class EarlyStopping:
    """
    source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=30, verbose=False, delta=1e-5, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.3
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        #self.trace_func = trace_func


    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     # if self.verbose:
    #         # self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), self.path)
    #     self.val_loss_min = val_loss

class WeightInitializer(object):
    """Random weights normalization and initialization"""
    def __call__(self, module):
        if hasattr(module, 'weight'):
            with torch.no_grad():
                # random initialization
                in_weights = np.random.uniform(0.03, 3, size=torch.numel(module.weight.data))
                # normalization
                in_weights /= in_weights.sum()
                #assignment
                module.weight.data = torch.from_numpy(in_weights.astype('float32'))


class WeightNormalizer(object):
    """Applied after each weight update, clips the weights to be in (0.01, 1) and normalises the sum to 1"""
    def __call__(self, module):
        if hasattr(module, 'weight'):
            weights = module.weight.data
            # weights clipping
            weights = weights.clamp(0.01,1)
            #weights normalization
            weights /= weights.sum() 
            #re-assignment
            module.weight.data = weights

def pytorch_linear_model(in_features=5, out_features=1):
    """Linear model builder no bias"""
    model = nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features, bias=False)
                              )
    return model