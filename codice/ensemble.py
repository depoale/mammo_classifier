import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy
from tools_for_Pytorch import EarlyStopping
import torch
from torch import nn
import warnings 
warnings.filterwarnings('ignore')

def epochs_array(array, max):
    epochs_count = np.linspace(0, max, len(array))
    return epochs_count

def train_ensemble(model, optimizer, normalizer, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=80,name=None):

    """Performs the forward and backwards training loop until early stopping, then computes the metric"""

    loss_fn = nn.MSELoss()
    acc_fn = BinaryAccuracy()
    early_stopping = EarlyStopping()

    torch.manual_seed(42)
    epochs = 500
    epoch_count = []

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
        for i in range(0,X_train.size()[0], batch_size):
            #optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            # train mode
            model.train()

            # 1. Forward pass on train data
            train_pred = model(batch_x)
            print('preds', train_pred)

            # 2. Calculate the loss and accuracy
            train_mse = loss_fn(train_pred, batch_y)
            train_acc = acc_fn(train_pred, batch_y)

            # append current batch results
            batch_mse.append(train_mse)
            batch_acc.append(train_acc)

            # 3. Zero grad of the optimizer
            optimizer.zero_grad()
            
            # 4. Backpropagation
            train_mse.backward()
            
            # 5. Progress the optimizer
            optimizer.step()

            # 6. Normalize new weights
            model.apply(normalizer)
            for param in model.parameters():
                print(param)
        
        # evaluation mode
        model.eval()
        
        # make predictions with model without gradient tracking 
        with torch.inference_mode():

            # 1. Forward pass on validation and test data (squeeze is required to adjust dimensions)
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
    if name: 
        plt.subplot(1,2,1)
        plt.plot(epoch_count, np.array(torch.tensor(train_mse_values).numpy()), label="Training MSE")
        plt.plot(epoch_count, val_mse_values, label="Validation MSE", linestyle='dashed')
        plt.plot(epoch_count, test_mse_values, label="External test MSE", linestyle='dotted')
        plt.title(name  + " TR and VL MSE")
        plt.ylabel("MSE")
        plt.xlabel("Epochs")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(epoch_count, np.array(torch.tensor(train_acc_values).numpy()), label="Training acc")
        plt.plot(epoch_count, val_acc_values, label="Validation acc", linestyle='dashed')
        plt.plot(epoch_count, test_acc_values, label="Extermal test acc", linestyle='dotted')
        plt.title(name  + " TR and VL acc")
        plt.ylabel("acc")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show(block = False)

    torch.save(model.state_dict(),'trained_ensemble.pt')
    return model.parameters(), final_acc, test_acc