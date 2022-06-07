import time
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data_functions import TrainData
import os, sys
import numpy as np
import random 
from torch.utils.tensorboard import SummaryWriter

from hinet_quant import HINet

import torch.quantization
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

from torch.quantization import QuantStub, DeQuantStub


##################### NOTE: Change the path to the dataset #####################
train_data_dir = "/mnt/d/DATASET/DATA_2070/train"
validate_data_dir = "/mnt/d/DATASET/DATA_2070/validate"
# test_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220325/test"
rain_L_dir = "rain_L"
rain_H_dir = "rain_H"
gt_dir = "gt"


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = HINet()

net_path = "../experiments/DeRain_512/models/hinet_naked.pth"
net.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))

# --- Load training data and validation/test data --- #
lbl_train_data_loader = DataLoader(TrainData([512, 512], train_data_dir, rain_L_dir, rain_H_dir, gt_dir), batch_size=1, shuffle=True, num_workers=4)

print("Number of training data: {}".format(len(lbl_train_data_loader)))

def preprocessImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image.astype(np.float32) / 255.0)
    return image

video_path = "/home/ao/tmp/clip_videos/h97cam_water_video.mp4"
video = cv2.VideoCapture(video_path)

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

input_img = preprocessImage(sample_image)
# input_img = input_img.to(device)


net.eval()
# net.train()
net.qconfig = torch.quantization.default_qconfig
# net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
net.fuse_model()

torch.quantization.prepare(net, inplace=True)
# torch.quantization.prepare_qat(net, inplace=True)

net(input_img)

net_quant = torch.quantization.convert(net, inplace=False)

print("[FINISHED] model conversion is finished.")

torch.onnx.export(net_quant, input_img, "../experiments/DeRain_512/models/hinet_pytorchPostTrainingQuant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=13)
# # torch.onnx.export(net, input_img, "./ckpt/transweather_pytorchPostTrainingQuant.onnx", verbose=True, export_params=True, do_constant_folding=True, input_names=['input'], output_names=['output'], opset_version=11)

print("[FINISHED] onnx model exported")



while True:
    ret, frame = video.read()
    if not ret:
        break
    sample_image = frame
    # sample_image = cv2.resize(frame, (960, 540))
    sample_image = cv2.resize(frame, (640, 360))
    sample_image = preprocessImage(sample_image).unsqueeze(0)
    print("---", sample_image.shape)
    pred = net_quant(sample_image)
    # pred = pred[0].detach().cpu().numpy()
    pred = pred[0].numpy()
    pred = pred * 255.0
    pred = pred.astype(np.uint8)
    pred = cv2.resize(pred, (640, 360))
    cv2.imshow("pred", pred)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # break



