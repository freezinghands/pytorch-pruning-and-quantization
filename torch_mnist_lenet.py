import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

# Setup warnings
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')
warnings.filterwarnings(action='default', module=r'torch.quantization')

model_name = "CNN_MNIST.pth"
model_path = os.path.join(os.curdir, 'torch_models')

'''
[1] Preparation of the datasets

Required interfaces
    training_data: defines training dataset
    test_data: defines test dataset
    train_dataloader: train dataset loader
    test_dataloader: test dataset loader
'''

dtransforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

# Download train data
training_data = datasets.MNIST(root="data",  # path to download the data
							   train=True,  # specifies the train or test dataset
							   download=True,  # download the data if not available at root
							   transform=dtransforms,  # transforms feature to tensor
							   )

# Download test data
test_data = datasets.MNIST(root="data", train=False, download=True, transform=dtransforms, )

batch_size = 64

# Create data loaders
# Data loader iterates with batch so it returns a batch with each iteration
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)  # prepare dataset for training
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)  # prepare dataset for testing

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


class LeNet5(nn.Module):
	def __init__(self):
		super(LeNet5, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
		self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
		self.fc1 = nn.Linear(120, 84)
		self.fc2 = nn.Linear(84, 10)

	def forward(self, x):  # method for forward propagation
		x = torch.nn.functional.tanh(self.conv1(x))
		x = torch.nn.functional.avg_pool2d(x, 2, 2)
		x = torch.nn.functional.tanh(self.conv2(x))
		x = torch.nn.functional.avg_pool2d(x, 2, 2)
		x = torch.nn.functional.tanh(self.conv3(x))
		x = x.view(-1, 120)
		x = torch.nn.functional.tanh(self.fc1(x))
		x = self.fc2(x)
		return torch.nn.functional.softmax(x, dim=1)


model = LeNet5().to(device)  # generate model
print(model)  # information about this model

for name, param in model.named_parameters():
	print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# Model optimization
# To train model you need loss and optimizer

loss_fn = nn.CrossEntropyLoss()  # loss functions
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # optimizer module


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

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
	print(model)
	# epochs = 20
	# for t in range(epochs):  # iterate for 'epochs' times
	# 	print(f"Epoch {t + 1}\n-------------------------------")
	# 	train(train_dataloader, model, loss_fn, optimizer)  # train the model
	# 	test(test_dataloader, model, loss_fn)  # test with current model to check current accuracy
	# print("Optimization completed")
	#
	# # saving the model
	# torch.save(model.state_dict(), os.path.join(model_path, model_name))
