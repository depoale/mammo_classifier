"""Ensemble training"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy
import torch
from torch import nn
import shutup
shutup.please()
import warnings 
warnings.filterwarnings('ignore')
from tools_for_Pytorch import EarlyStopping


def train_ensemble(model, optimizer, normalizer, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=80):
    """Performs the forward and backward training loop until early stopping, then computes the metric.
    ...
    Parameters
    ---------
    model: pytorch model
        linear ensemble model
    optimizer: torch.optim
    normalizer: WeightNormalizer
        applies clipping and normalisation after each optimiser step
    X_train: torch.tensor
    y_train: torch.tensor
    X_val: torch.tensor
    y_val: torch.tensor
    X_test: torch.tensor
    y_test: torch.tensor
    batch_size: int
        Default 80
    
    Returns
    -------
    weights: torch.tensor
        ensemble trained weights
    final_acc: list
    test_acc: list
    """

    loss_fn = nn.MSELoss()
    acc_fn = BinaryAccuracy()
    early_stopping = EarlyStopping()

    torch.manual_seed(42)
    epochs = 500    #artificially large number of epochs (expected to stop with early stopping)
    epoch_count = []

    # initialising lists to keep track of the performance
    train_mse_values = []
    val_mse_values = []
    test_mse_values = []
    train_acc_values = []
    val_acc_values = []
    test_acc_values = []

    for epoch in range(epochs):

        #shuffle before creating mini-batches
        permutation = torch.randperm(X_train.size()[0])
        batch_mse=[]
        batch_acc = []
        #one training loop for each mini batch
        for i in range(0, X_train.size()[0], batch_size):
            
            # 1. Create mini-batch
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            # 2. Train mode
            model.train()

            # 3. Forward pass on train batch data
            train_pred = model(batch_x)

            # 4. Calculate the loss and accuracy
            train_mse = loss_fn(train_pred, batch_y)
            train_acc = acc_fn(train_pred, batch_y)

            # 5. append current batch results
            batch_mse.append(train_mse)
            batch_acc.append(train_acc)

            # 6. Zero grad of the optimizer
            optimizer.zero_grad()
            
            # 7. Backpropagation
            train_mse.backward()
            
            # 8. Progress the optimizer
            optimizer.step()

            # 9. Normalize new weights
            model.apply(normalizer)

        # evaluation mode
        model.eval()
        
        # make predictions with model without gradient tracking 
        with torch.inference_mode():

            # 1. Forward pass on validation and test data 
            val_pred = torch.squeeze(model(X_val))
            test_pred = torch.squeeze(model(X_test))

            # 2. Caculate loss and acc on validation and test data        
            val_mse = loss_fn(val_pred, y_val)                    
            test_mse = loss_fn(test_pred, y_test)
            val_acc = acc_fn(val_pred, y_val)                    
            test_acc = acc_fn(test_pred, y_test)

        # append current epoch results
        epoch_count.append(epoch)
        train_mse_values.append(np.mean(np.array(torch.tensor(batch_mse).numpy())))
        val_mse_values.append(val_mse)
        test_mse_values.append(test_mse)
        train_acc_values.append(np.mean(np.array(torch.tensor(batch_acc).numpy())))
        val_acc_values.append(val_acc)
        test_acc_values.append(test_acc)

    
        # early_stopping needs the validation loss to check if it has decreased
        early_stopping(val_mse, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
        if epoch % 10 == 0:
            print(f"Epoch is {epoch:<3} | Training MSE: {train_mse:.3f} | Validation MSE: {val_mse:.3f} | Training acc: {train_acc:.3f} | Validation acc: {val_acc:.3f}")

    print(f"Epoch is {epoch:<3} \nTraining MSE: {train_mse:.3f} | Validation MSE: {val_mse:.3f} | Test MSE: {test_mse:.3f} | Training acc: {train_acc:.3f} | Validation acc: {val_acc:.3f}")
   
    final_acc = val_acc_values[-1]
    test_acc = test_acc_values[-1]

    #learning curve and accuracy plot
    plt.subplot(1,2,1)
    plt.plot(epoch_count, np.array(torch.tensor(train_mse_values).numpy()), label="Training MSE")
    plt.plot(epoch_count, val_mse_values, label="Validation MSE", linestyle='dashed')
    plt.plot(epoch_count, test_mse_values, label="External test MSE", linestyle='dotted')
    plt.title("Ensemble TR, VL and TS MSE")
    plt.ylabel("MSE")
    plt.xlabel("Epochs")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epoch_count, np.array(torch.tensor(train_acc_values).numpy()), label="Training acc")
    plt.plot(epoch_count, val_acc_values, label="Validation acc", linestyle='dashed')
    plt.plot(epoch_count, test_acc_values, label="Extermal test acc", linestyle='dotted')
    plt.title("Ensemble TR, VL and TS acc")
    plt.ylabel("acc")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show(block = False)

    #save ensemble weights
    torch.save(model.state_dict(),'trained_ensemble.pt')
    return model[0].weight.data, final_acc, test_acc