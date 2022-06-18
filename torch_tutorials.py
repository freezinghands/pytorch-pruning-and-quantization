import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.quantization.quantize_fx as quantize_fx
import torch.nn.utils.prune as prune

import copy
import matplotlib.pyplot as plt


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


# # See images for 9 random target labels
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()  # get random sample index
#     img, label = training_data[sample_idx]  # get image and label (can iterate just like list)
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


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
'''

# Creating models
# In pytorch, model inherits nn.Module
# Structure of the model is defined inside the '__init__' method
# Forward propagation of the model is defined in 'forward' method

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model_path = os.path.join(os.curdir, 'torch_models')
model_name = "MLP_FashionMNIST"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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

model = NeuralNetwork().to(device)  # generate model
print(model)  # information about this model


# Even though forward propagation is defined inside the 'forward' method,
# do not call 'forward' method directly just to call forward propagation of the model
# Pass an input data directly to the model just like a function instead

# X = torch.rand(1, 28, 28, device=device)  # random generation of the input data
# logits = model(X)                         # forward propagation
# pred_probab = nn.Softmax(dim=1)(logits)   # predicted result with softmax layer (probability vector)
# y_pred = pred_probab.argmax(1)            # get index of maximum argument
# print(f"Predicted class: {y_pred}")


# Model parameters
# You can simply extract each hyperparameters of each layers inside the model
# The code below simply prints each hyperparameter of the layers inside the model if there's any parameter

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


epochs = 5
for t in range(epochs):                                 # iterate for 'epochs' times
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)  # train the model
    test(test_dataloader, model, loss_fn)               # test with current model to check current accuracy
print("Optimization completed")


'''
[3] Quantization examples
'''

# FX Graph mode quantization samples

model_to_quantize = copy.deepcopy(model)                     # copy model for quantization
model_to_quantize.eval()                                     # set model into evaluation mode
qconfig = torch.quantization.get_default_qconfig('qnnpack')  # set Qconfig
qconfig_dict = {"": qconfig}                                 # generate Qconfig

def calibrate(model, data_loader):         # calibration function
    model.eval()                           # set to evaluation mode
    with torch.no_grad():                  # do not save gradient when evaluation mode
        for image, target in data_loader:  # extract input and output data
            model(image)                   # forward propagation

model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)  # preparation
calibrate(model_prepared, test_dataloader)                                # calibration
model_quantized = quantize_fx.convert_fx(model_prepared)                  # convert the model

print(model_quantized)


'''
[4] Pruning examples
'''

prune.random_unstructured(model.lin1, "weight", amount=0.3)
prune.random_unstructured(model.lin1, "bias",   amount=0.3)
prune.random_unstructured(model.lin2, "weight", amount=0.3)
prune.random_unstructured(model.lin2, "bias",   amount=0.3)
prune.random_unstructured(model.lin3, "weight", amount=0.3)
prune.random_unstructured(model.lin3, "bias",   amount=0.3)

print("pruning completed")


'''
[5] Extracting intermediate activations with forward hook
See more about this tutorial: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/05/27/extracting-features.html
See more about hooks: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
'''

output_dirname = os.path.join(os.curdir, "torch_model_outputs")
features = {}

def make_hook(name):
    def extract_output(model, model_input, model_output):
        features[f"{name}_output"] = model_output.detach()
    return extract_output


model_quantized.flatten.register_forward_hook(make_hook(f'{model_name}_flatten'))
model_quantized.lin1.register_forward_hook(make_hook(f'{model_name}_lin1'))
model_quantized.lin2.register_forward_hook(make_hook(f'{model_name}_lin2'))
model_quantized.lin3.register_forward_hook(make_hook(f'{model_name}_lin3'))

test(test_dataloader, model_quantized, loss_fn)

print(features.keys())

for layer_name in features.keys():
    torch.save(features[layer_name], os.path.join(output_dirname, f"{layer_name}"))