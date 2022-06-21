# CLASS INDEX FILE(.txt) FROM https://github.com/pytorch/hub/blob/master/imagenet_classes.txt


import os
import torch
import torchvision
import pandas as pd
from torch import nn
import math

model_name = 'Mobilenetv2.pth'
model_path = os.path.join(os.curdir, 'torch_models')
loss_fn = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class CustomImageDataset(torch.utils.data.Dataset):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
		self.img_labels = pd.read_csv(annotations_file)
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
		# image = torchvision.io.read_image(img_path)
		image = torchvision.io.read_image(img_path, mode=torchvision.io.image.ImageReadMode(torchvision.io.ImageReadMode.RGB))
		label = self.img_labels.iloc[idx, 1]
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		if image.size()[0] == 1:
			image = torch.cat([image[0]] * 3, dim=0)
			image.resize_(3, 64, 64)
		return image, label


def test(dataloader, model, loss_fn):
	size = len(dataloader.dataset)  # dataset size
	num_batches = len(dataloader)  # the number of batches
	model.eval()  # convert model into evaluation mode
	test_loss, correct = 0, 0  # check total loss and count correctness
	with torch.no_grad():  # set all of the gradient into zero
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)  # extract input and output
			pred = model(X.float())  # predict with the given model # fixed
			test_loss += loss_fn(pred, y).item()  # acculmulate total loss
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # count correctness
	test_loss /= num_batches  # make an average of the total loss
	correct /= size  # make an average with correctness count
	print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

img_labels = pd.read_csv("imagenet_class_labels_train.csv")
# for n in range(10000):
# 	img_path = os.path.join("../../tiny-imagenet-200-trainset", img_labels.iloc[n, 0])
# 	image = torchvision.io.read_image(img_path)
# 	if image.size()[0] < 3:
# 		print(f"n = {n + 2}, {image.size()}")
# 		image *= 3
# 	# print(image)
# image_path_n01774750_114 = os.path.join("../../tiny-imagenet-200-trainset", img_labels.iloc[8387, 0])
# image = torchvision.io.read_image(image_path_n01774750_114)
# print(f"\nfilename: {img_labels.iloc[8387, 0]}, {image.size()} -> ", end='')
# # image_resize = []
# # for i in range(3):
# # 	tensor = image[0]
# # 	image_resize.append(tensor)
# image = torch.cat([image[0]]*3, dim=0)
# image.resize_(3, 64, 64)
# print(image.size())
# print(image.size()[0])



def collate_fn(batch):
	batch = list(filter(lambda x: x is not None, batch))
	return torch.utils.data.dataloader.default_collate(batch)

mobilenet_transform = torchvision.transforms.Compose([
	# torchvision.transforms.ToTensor(),
	torchvision.transforms.Resize((32, 32)),
	# torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

training_data = CustomImageDataset('imagenet_class_labels_train_truncated.csv', '../../tiny-imagenet-200-trainset', transform=mobilenet_transform)
test_data = CustomImageDataset('imagenet_class_labels_train_truncated.csv', '../../tiny-imagenet-200-trainset', transform=mobilenet_transform)

batch_size = 64
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# x_train, y_train = training_data.data, training_data.targets

for X, y in test_dataloader:
	print(f"Shape of X [N, C, H, W]: {X.shape} {X[0][0][0][0].dtype}")   # (3, 64, 64), dtype=torch.uint8
	print(f"Shape of y: {y.shape} {y.dtype}")                    # ----------------> [64] - this 64 is batch_size
	print(f"y: {y}")
	break



'''
# We need only Instance so we skip dataset fetching
# REFERENCE : https://github.com/d-li14/mobilenetv2.pytorch
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
'''

if __name__ == '__main__':
	model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
	for name, param in model.named_parameters():
		print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
	# torch.save(model.state_dict(), os.path.join(model_path, model_name))
	# print(f"model {model_name.split('.')[0]} successfully saved.")
	# print(model)
	# x_test, y_test = test_dataloader
	# x_test, y_test = test_dataloader
	# print(x_test)
	# print(y_test)

	# model.eval()
	# params = model.state_dict()
	# dummy_input = torch.empty(1, 3, 64, 64, dtype=torch.float32)
	# torch.onnx.export(model, dummy_input, "Mobilenetv2.onnx")   #, export_params=True,
	# 				  # input_names=["modelInput"], output_names=["modelOutput"],
	# 				  # dynamic_axes={'modelInput': {0: 'batch_size'}, 'modelOutput': {0: 'batch_size'}})
	# print("Convert to onnx successfully.")

# if __name__ == '__main__':
# 	model = MobileNetV2()
# 	# print(model)
# 	# model.load_state_dict(torch.load(os.path.join(model_path, model_name), map_location='cpu'))  # load save state_dict
# 	print(model)


