import os
import torch
import torchvision
import onnx


model_list = ['fcn_resnet50', #0
              'fcn_resnet101',#1
              'deeplabv3_resnet50', #2
              'deeplabv3_resnet101', #3 
              'deeplabv3_mobilenet_v3_large', #4
              'lraspp_mobilenet_v3_large' #5
             ]

def fcn_resnet50():
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, aux_loss=False).cuda()
    return model

def fcn_resnet101():
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=True, aux_loss=False).cuda()
    return model

def deeplabv3_resnet50():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, aux_loss=False).cuda()
    return model

def deeplabv3_resnet101():
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, aux_loss=False).cuda()
    return model

def deeplabv3_mobilenet_v3_large():
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, aux_loss=False).cuda()
    return model

def lraspp_mobilenet_v3_large():
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True, aux_loss=False).cuda()
    return model

def default():
    print('Please choose the right model index')
    print(model_list)
    return None

choose_model = {'fcn_resnet50': fcn_resnet50, #0
                'fcn_resnet101': fcn_resnet101, #1
                'deeplabv3_resnet50': deeplabv3_resnet50, #2
                'deeplabv3_resnet101': deeplabv3_resnet101, #3
                'deeplabv3_mobilenet_v3_large': deeplabv3_mobilenet_v3_large, #4
                'lraspp_mobilenet_v3_large': lraspp_mobilenet_v3_large #5
               }

mdoel_index = 2
model_name = model_list[mdoel_index]
model = choose_model.get(model_name, default)()
# print(model)


batch_size = 1
color_channel = 3
height = 256
width = 256

input_names = ['input0'] + ['learned_%d' %i for i in range(16)]
output_names = ['dense_out']
dummy_input = torch.randn(batch_size, color_channel, width, height, device='cuda')
onnx_model_name = 'onnx_models/'+ model_name + '.onnx'
dynamic_axes = {'input0': {0:'batch', 2:'height', 3:'width'}}

torch.onnx.export(model,
                 dummy_input,
                 onnx_model_name,
                 input_names=input_names,
                 output_names=output_names,
                 opset_version=11,
                 dynamic_axes=dynamic_axes)
                 
model = onnx.load(onnx_model_name)
info = onnx.checker.check_model(onnx_model_name)
print(info)

