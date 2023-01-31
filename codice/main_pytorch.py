import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import read_imgs
import ensemble_pytorch as t
import keras
from tools_for_Pytorch import get_predictions

X_train, y_train = read_imgs('data_png/Train', [0,1])
X_test, y_test = read_imgs('data_png/Test', [0,1])
model1=keras.models.load_model('model1')
model2=keras.models.load_model('model2')
model3=keras.models.load_model('model3')
models_list = [model1, model2, model3]
y = get_predictions(X_train, models_list)
print(y.shape)

X = torch.from_numpy(X_train)
y = torch.from_numpy(y_train)
train_dataloader = DataLoader(X,y, batch_size=64, shuffle=True)
num_epochs = 15
model_densenet161 = models.densenet161(pretrained=True)
for param in model_densenet161.parameters():
    param.requires_grad = False
model_densenet161.classifier = torch.nn.Linear(model_densenet161.classifier.in_features, out_features=200)
model_densenet161 = model_densenet161.to(t.DEVICE)

densenet161_training_results = t.training(model=model_densenet161,
                                        model_name='DenseNet161',
                                        num_epochs=num_epochs,
                                        train_dataloader=train_dataloader,
                                    )

model_densenet161, train_loss_array, train_acc_array, val_loss_array, val_acc_array = densenet161_training_results

min_loss = min(val_loss_array)
min_loss_epoch = val_loss_array.index(min_loss)
min_loss_accuracy = val_acc_array[min_loss_epoch]

t.visualize_training_results(train_loss_array,
                           val_loss_array,
                           train_acc_array,
                           val_acc_array,
                           num_epochs,
                           model_name="DenseNet161",
                           batch_size=64)
print("\nTraining results:")
print("\tMin val loss {:.4f} was achieved during epoch #{}".format(min_loss, min_loss_epoch + 1))
print("\tVal accuracy during min val loss is {:.4f}".format(min_loss_accuracy))