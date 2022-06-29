import os

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision.transforms import ToTensor

import torch.quantization.quantize_fx as quantize_fx
import torch.nn.utils.prune as prune

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

dataset_name = "TinyImageNet"
# target_dataset = datasets.STL10  # target dataset
target_dataset = datasets.ImageFolder  # target dataset--custom dataset    dir tree: train, val
train_batch_size = 128  # batch size (for further testing of the model)
test_batch_size = 64    # batch size (for further testing of the model)


train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.ToTensor(),])
test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor(),])

# # Download train data
# training_data = target_dataset(
#     root='data/tiny-imagenet/train',  # path to download the data
#     # split='train',
#     # download=True,  # download the data if not available at root
#     transform=ToTensor(),  # transforms feature to tensor
# )

# # Download test data
# test_data = target_dataset(
#     root='data/tiny-imagenet',
#     # split='test',
#     # download=True,
#     transform=ToTensor(),
# )
# Prepare whole (train, test) data
whole_data = target_dataset(
    root='data/tiny-imagenet/train',
    transform=ToTensor(),
)

training_data, test_data = random_split()

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)  # prepare dataset for training
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)  # prepare dataset for testing

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
""""""
print(training_data.class_to_idx)
print(test_data.class_to_idx)
print(type(training_data))
print(type(test_data))
train_dataiter, test_dataiter = iter(train_dataloader), iter(test_dataloader)
train_imgs, train_labels = next(train_dataiter)
test_imgs, test_labels = next(test_dataiter)
print(train_imgs.size())
print(test_imgs.size())
print(train_labels)
print(train_labels.size())
print(test_labels)
print(test_labels.size())
input("Stop point.")
""""""

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
# print(model)  # information about this model

# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

print(model)

# Model optimization
# To train model you need loss and optimizer
# lr = 0.0001
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer module
loss_fn = nn.CrossEntropyLoss()  # loss functions


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


def test(dataloader, model, loss_fn, max_iter=None):
    size = len(dataloader.dataset)  # dataset size
    num_batches = len(dataloader)  # the number of batches
    model.eval()  # convert model into evaluation mode
    test_loss, correct = 0, 0  # check total loss and count correctness
    iter_cnt = 0
    with torch.no_grad():  # set all of the gradient into zero
        for X, y in dataloader:
            if iter_cnt > max_iter:
                break
            else:
                iter_cnt += 1
            X, y = X.to(device), y.to(device)  # extract input and output
            pred = model(X)  # predict with the given model
            test_loss += loss_fn(pred, y).item()  # acculmulate total loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # count correctness
            print(f'\rtest iter: {iter_cnt}', end='')
    print()
    test_loss /= num_batches  # make an average of the total loss
    correct /= size  # make an average with correctness count
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# optimization of the model
# if __name__ == '__main__':
#     epochs = 5
#     for t in range(epochs):  # iterate for 'epochs' times
#         print(f"Epoch {t + 1}\n-------------------------------")
#         train(train_dataloader, model, loss_fn, optimizer)  # train the model
#         test(test_dataloader, model, loss_fn)  # test with current model to check current accuracy
#     print("Optimization completed")


prune_amount = 0.7  # pruning amount

def prune_layer(layer_module, max_sublayer_idx):
    prune.l1_unstructured(layer_module[0].downsample[0], 'weight', amount=prune_amount)
    prune.remove(layer_module[0].downsample[0], 'weight')

    for idx in range(max_sublayer_idx+1):
        prune_sublayer_conv2d(layer_module, idx)

def prune_sublayer_conv2d(layer_module, idx):
    prune.l1_unstructured(layer_module[idx].conv1, 'weight', amount=prune_amount)
    prune.l1_unstructured(layer_module[idx].conv2, 'weight', amount=prune_amount)
    prune.l1_unstructured(layer_module[idx].conv3, 'weight', amount=prune_amount)

    prune.remove(layer_module[idx].conv1, 'weight')
    prune.remove(layer_module[idx].conv2, 'weight')
    prune.remove(layer_module[idx].conv3, 'weight')

if __name__ == '__main__':
    prune.l1_unstructured(model.conv1, 'weight', amount=prune_amount)
    prune.remove(model.conv1, 'weight')

    prune_layer(model.layer1, 2)
    prune_layer(model.layer2, 3)
    prune_layer(model.layer3, 5)
    prune_layer(model.layer4, 2)

    prune.l1_unstructured(model.fc, 'weight', amount=prune_amount)
    prune.l1_unstructured(model.fc, 'bias', amount=prune_amount)
    prune.remove(model.fc, 'weight')
    prune.remove(model.fc, 'bias')

    print("pruning completed")

# isinstance
# eval, exec
# _modules[]
#
# def prune_all_layers(module: torch.nn.Module, prune_amount):
#     for sub_module in module._modules:
#         if isinstance(sub_module, torch.nn.Module):
#             prune_all_layers(sub_module, prune_amount)



quant_type = 'static'
model.eval()                                                 # set model into evaluation mode
qconfig = torch.quantization.get_default_qconfig('fbgemm')  # set Qconfig
qconfig_dict = {"": qconfig}                                 # generate Qconfig

def calibrate(model, data_loader):         # calibration function
    cnt = 1
    model.eval()                           # set to evaluation mode
    with torch.no_grad():                  # do not save gradient when evaluation mode
        for image, target in data_loader:  # extract input and output data
            model(image)                   # forward propagation
            print(f'\rcalibration iter: {cnt:3d}/{len(data_loader):3d}', end='')
            cnt += 1
    print()

if __name__ == '__main__':
    model_prepared = quantize_fx.prepare_fx(model, qconfig_dict)  # preparation
    calibrate(model_prepared, test_dataloader)                    # calibration
    model_quantized = quantize_fx.convert_fx(model_prepared)      # convert the model

    print('quantization completed')
    print(model_quantized)


'''
[3] Saving generated models

Required interfaces
    model_name: name of the model (state_dict)
    model_path: path toward the saved model (state_dict)

Note
    There are two ways to save model
    1) Saving with state_dict: only saves parameters of the given model
    2) Saving with pickle: actually saves python pickle of the given model

    This source code saves model with 2) by default
'''

model_name = f"{model_type}_{dataset_name}_{quant_type}_{int(prune_amount*100)}.pth"
model_path = os.path.join(os.curdir, 'torch_models')

if __name__ == '__main__':
    # saving the model
    torch.save(model_quantized.state_dict(), os.path.join(model_path, model_name))
    # torch.save(model, os.path.join(model_path, model_name))
    print('model saved')