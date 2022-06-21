import os
import torch
import torch.quantization.quantize_fx as quantize_fx
import torch.nn.utils.prune as prune
import pandas as pd


'''
[1] Model Setup Section
'''

import torch_mobilenetv2 as modelfile  # modelfile selection

model_path = modelfile.model_path  # path toward saved target model
model_name = modelfile.model_name  # target model name

test_dataloader = modelfile.test_dataloader  # test dataset
test = modelfile.test                        # test function
loss_fn = modelfile.loss_fn                  # loss function

# model = modelfile.MobileNetV2()  # generate model of modelfile
# model.load_state_dict(torch.load(os.path.join(model_path, model_name)))  # load save state_dict
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)


'''
[2] Pruning Section
'''

prune_amount = 0.0  # pruning amount

# Define layers to prune
prune.l1_unstructured(model.features[0][0], "weight", amount=prune_amount)
# prune.l1_unstructured(model.features[18][0], "weight", amount=prune_amount)

# print(model.features[0][0].weight)

input()

print("pruning completed")


'''
[3] Quantization Section
'''

# model_to_quantize = copy.deepcopy(model)  # copy model for quantization
model.eval()                                                 # set model into evaluation mode
qconfig = torch.quantization.get_default_qconfig('qnnpack')  # set Qconfig
qconfig_dict = {"": qconfig}                                 # generate Qconfig

def calibrate(model, data_loader):         # calibration function
    model.eval()                           # set to evaluation mode
    with torch.no_grad():                  # do not save gradient when evaluation mode
        for image, target in data_loader:  # extract input and output data
            model(image)                   # forward propagation

model_prepared = quantize_fx.prepare_fx(model, qconfig_dict)  # preparation
# calibrate(model_prepared, test_dataloader)                    # calibration
model_quantized = quantize_fx.convert_fx(model_prepared)      # convert the model

print(model_quantized)


'''
[4] Model output and parameter extraction
'''

# Setup model for extracting data
target_model = model_quantized
output_dirname = os.path.join(os.curdir, "torch_model_outputs")  # path toward saved data
model_info = f'quant_static_prune_{int(prune_amount * 100)}'     # model information (quant and pruning)
output_modelname = f"{model_name.split('.')[0]}_{model_info}"    # model name for output data

features = {}             # dictionary containing extracted data
activation_cnt_limit = 5  # the number of intermidiate activation to save

# Define specific action while extracting data
def make_hook(name):
    def extract_output(model, model_input, model_output):
        # Calculate activation count
        activation_cnt = 1
        while f"{name}_output{activation_cnt}" in features: activation_cnt += 1
        if activation_cnt > activation_cnt_limit: return

        # Extract output data
        print(f"extracting {name}_output{activation_cnt}")
        features[f"{name}_output{activation_cnt}"] = model_output.detach()
    return extract_output

# Define layers to extract output
target_model.features.register_forward_hook(make_hook(f'{output_modelname}_features'))
target_model.classifier.register_forward_hook(make_hook(f'{output_modelname}_classifier'))
# target_model.conv3.register_forward_hook(make_hook(f'{output_modelname}_conv3'))
# target_model.fc1.register_forward_hook(make_hook(f'{output_modelname}_fc1'))
# target_model.fc2.register_forward_hook(make_hook(f'{output_modelname}_fc2'))

# Forward propagation by using test dataset provided by modelfile
test(test_dataloader, target_model, loss_fn)

# Extracting parameters
for param_name in target_model.state_dict():
    if '._packed_params._packed_params' in param_name:
        print(f"extracting {output_modelname}_{param_name.split('.')[0]}_weight")
        features[f"{output_modelname}_{param_name.split('.')[0]}_weight"] = target_model.state_dict()[param_name][0].int_repr().detach()

print(f"\n{len(features)} data extracted!")

# Saving extracted data
for layer_name in features.keys():
    torch.save(features[layer_name], os.path.join(output_dirname, f"{layer_name}"))