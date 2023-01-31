import numpy as np
from torchmetrics import Metric

class EarlyStopping:
    """
    source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=20, verbose=False, delta=0.3, path='checkpoint.pt', trace_func=print):
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


def weights_init_uniform_fan_in(m):
    '''Initizialize weights with a uniform distribution, according to fan-in rule, and set bias to 0'''
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        fan_in = m.in_features
        limit = np.sqrt(6 / fan_in)
        m.weight.data.uniform_(-limit, limit)
        m.bias.data.fill_(0)

def count_parameters(model):
    '''Number of trainable parameters '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_predictions(X, experts):
    y = np.empty(shape=(len(X), len(experts)))
    for count, expert in enumerate(experts):
        y[:, count] = np.squeeze(expert.predict(X))
    return y
    

