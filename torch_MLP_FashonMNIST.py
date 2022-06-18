import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
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

# Download train data
training_data = datasets.FashionMNIST(
    root="data",  # path to download the data
    train=True,  # specifies the train or test dataset
    download=True,  # download the data if not available at root
    transform=ToTensor(),  # transforms feature to tensor
)

# Download test data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders
# Data loader iterates with batch so it returns a batch with each iteration
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)  # prepare dataset for training
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)       # prepare dataset for testing

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

class NetworkModel(nn.Module):
    def __init__(self):
        super(NetworkModel, self).__init__()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(28 * 28, 512)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(512, 512)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(512, 10)

    def forward(self, x):  # method for forward propagation
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        logits = self.lin3(x)
        return logits

model = NetworkModel().to(device)  # generate model
print(model)  # information about this model

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# Model optimization
# To train model you need loss and optimizer

loss_fn = nn.CrossEntropyLoss()  # loss functions
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # optimizer module

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # size of the dataset
    model.train()                   # turn the model into train mode
    for batch, (X, y) in enumerate(dataloader):  # each index of dataloader will be batch index
        X, y = X.to(device), y.to(device)        # extract input and output

        # Compute prediction error
        pred = model(X)          # predict model
        loss = loss_fn(pred, y)  # calculate loss

        # Backpropagation
        optimizer.zero_grad()  # gradient initialization (just because torch accumulates gradient)
        loss.backward()        # backward propagate with the loss value (or vector)
        optimizer.step()       # update parameters

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # dataset size
    num_batches = len(dataloader)   # the number of batches
    model.eval()                    # convert model into evaluation mode
    test_loss, correct = 0, 0       # check total loss and count correctness
    with torch.no_grad():           # set all of the gradient into zero
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)     # extract input and output
            pred = model(X)                       # predict with the given model
            test_loss += loss_fn(pred, y).item()  # acculmulate total loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # count correctness
    test_loss /= num_batches   # make an average of the total loss
    correct /= size            # make an average with correctness count
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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

model_name = "MLP_FashionMNIST.pth"
model_path = os.path.join(os.curdir, 'torch_models')

# saving the model
if __name__ == '__main__':
    torch.save(model.state_dict(), os.path.join(model_path, model_name))