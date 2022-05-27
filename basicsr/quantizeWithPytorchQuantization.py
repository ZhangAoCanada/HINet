import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from tabnanny import verbose

import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from train_data_functions import TrainData
import os
import numpy as np
import random
from hinet_custom import HINet

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from random import randrange
import torchvision.utils as utils
import cv2
import re
from tqdm import tqdm
from skimage import img_as_ubyte

from torchinfo import summary

### NOTE: for quantization ###
import torchvision
from torchvision import transforms
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from absl import logging
logging.set_verbosity(logging.FATAL)

quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

from pytorch_quantization import quant_modules
quant_modules.initialize()


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image.astype(np.float32) / 255.0)
    return image


def postprocess(pred):
    pred = pred[0].float().detach().cpu().numpy()
    pred = (pred * 255.).round()
    pred = pred.astype(np.uint8)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    return pred


### NOTE: create data loader ###
# train_data_dir = "/mnt/d/DATASET/DATA_2070/test/"
train_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220325/train"
rain_L_dir = "rain_L"
rain_H_dir = "rain_H"
gt_dir = "gt"
crop_size = [512, 512]
batch_size = 1
data_loader = DataLoader(TrainData(crop_size, train_data_dir, rain_L_dir, rain_H_dir, gt_dir), batch_size=batch_size, shuffle=True, num_workers=4)

# video_path = "/home/ao/tmp/clip_videos/videos/dusty_video1.mp4"
video_path = "/content/drive/MyDrive/DERAIN/video_data/h97cam_water_video.mp4"
model_path = "../experiments/DeRain_512/models/hinet_naked.pth"

video = cv2.VideoCapture(video_path)

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

net = HINet()

if device == torch.device("cpu"):
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("====> model ", model_path, " loaded")
else:
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    print("====> model ", model_path, " loaded")

# net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# print("====> model ", model_path, " loaded")

net.eval()

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        # model(image.cuda())
        model(image)
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
            
def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    # model.cuda()

with torch.no_grad():
    collect_stats(net, data_loader, num_batches=2)
    # compute_amax(net, method='percentile', percentile=99.99)
    compute_amax(net, method='entropy')


torch.save(net.state_dict(), "../experiments/DeRain_512/models/quantized_model.pth")

print("[INFO] quantization .pt model saved.")


sample_img = None
while True:
    ret, frame = video.read()
    if not ret:
        break
    sample_image = frame
    sample_image = cv2.resize(frame, (640, 480))
    break

if sample_image is not None:
    print("[INFO] image shape: ", sample_image.shape)
else:
    print("[INFO] image is None")


input_img = sample_image
input_img = preprocess(input_img)

torch.onnx.export(net, input_img, "../experiments/DeRain_512/models/transweather_quant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=13)
# torch.onnx.export(net, input_img, "./ckpt/transweather_quant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=14, enable_onnx_checker=False)
# torch.onnx.export(net, input_img, "./ckpt/transweather_quant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=13, enable_onnx_checker=False,dynamic_axes={'input': {0, 'batch_size'}, 'output': {0, 'batch_size'}})

print("[FINISHED] onnx model exported")

