import os

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision.transforms import ToTensor

# Setup warnings
import warnings

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)


'''
[1] Preparation of the datasets

Required interfaces
    training_data: defines training dataset
    test_data: defines test dataset
    train_dataloader: train dataset loader
    test_dataloader: test dataset loader
'''

# Setup for dataset
dataset_name = "STL10"
target_dataset = datasets.STL10  # target dataset
train_batch_size = 128           # batch size (for further testing of the model)
test_batch_size = 64             # batch size (for further testing of the model)


train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.ToTensor(),])
test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor(),])

# Download train data
training_data = target_dataset(
    root="data",  # path to download the data
    split='train',
    download=True,  # download the data if not available at root
    transform=ToTensor(),  # transforms feature to tensor
)

# Download test data
test_data = target_dataset(
    root="data",
    split='test',
    download=True,
    transform=ToTensor(),
)

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)  # prepare dataset for training
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)  # prepare dataset for testing

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


'''
[2] Creation of the model

Required interfaces
    NetworkModel: structure of the model
    train: training function
    test: test function

Note
    Creation and optimization of the model only occurrs when this code is call as 'main'
'''

# Creating models
# In pytorch, model inherits nn.Module
# Structure of the model is defined inside the '__init__' method
# Forward propagation of the model is defined in 'forward' method

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Setup for model
model_type = "resnet50"
NetworkModel = models.resnet50

# Generate model for fine tuning
model = NetworkModel(pretrained=True).to(device)  # generate model with pretrained weight
print(model)  # information about this model

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# Model optimization
# To train model you need loss and optimizer
lr = 0.0001
loss_fn = nn.CrossEntropyLoss()  # loss functions
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer module


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # size of the dataset
    model.train()  # turn the model into train mode
    for batch, (X, y) in enumerate(dataloader):  # each index of dataloader will be batch index
        X, y = X.to(device), y.to(device)  # extract input and output

        # Compute prediction error
        pred = model(X)  # predict model
        loss = loss_fn(pred, y)  # calculate loss

        # Backpropagation
        optimizer.zero_grad()  # gradient initialization (just because torch accumulates gradient)
        loss.backward()  # backward propagate with the loss value (or vector)
        optimizer.step()  # update parameters

        loss, current = loss.item(), batch * len(X)
        print(f"batch idx: {batch}  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # dataset size
    num_batches = len(dataloader)  # the number of batches
    model.eval()  # convert model into evaluation mode
    test_loss, correct = 0, 0  # check total loss and count correctness
    with torch.no_grad():  # set all of the gradient into zero
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # extract input and output
            pred = model(X)  # predict with the given model
            test_loss += loss_fn(pred, y).item()  # acculmulate total loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # count correctness
    test_loss /= num_batches  # make an average of the total loss
    correct /= size  # make an average with correctness count
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# optimization of the model
if __name__ == '__main__':
    epochs = 5
    for t in range(epochs):  # iterate for 'epochs' times
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)  # train the model
        test(test_dataloader, model, loss_fn)  # test with current model to check current accuracy
    print("Optimization completed")


'''
[3] Saving generated models

Required interfaces
    model_name: name of the model (state_dict)
    model_path: path toward the saved model (state_dict)

Note
    There are two ways to save model
    1) Saving with state_dict: only saves parameters of the given model
    2) Saving with pickle: actually saves python pickle of the given model

    This source code saves model with 1) by default
'''

model_name = f"{model_type}_{dataset_name}.pth"
model_path = os.path.join(os.curdir, 'torch_models')

# saving the model
if __name__ == '__main__':
    torch.save(model.state_dict(), os.path.join(model_path, model_name))