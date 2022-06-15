# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import sys

from cv2 import VideoWriter
sys.path.append("/content/drive/MyDrive/DERAIN/HINet")

import importlib
import logging
from unittest import TestLoader
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
from basicsr.utils import get_root_logger, imwrite 
from tqdm import tqdm
from copy import deepcopy
import time, cv2
from skimage.measure import compare_psnr, compare_ssim
import numpy as np
from torchvision.utils import make_grid
import math

from hinet_custom import HINet

# from google.colab.patches import cv2_imshow


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


def main():
    # video_path = "/home/ao/tmp/clip_videos/videos/dusty_video1.mp4"
    video_path = "/content/drive/MyDrive/DERAIN/DATA_captured/something_else/dust_with_water2_video.mp4"
    video = cv2.VideoCapture(video_path)

    net = HINet()
    net_path = "../experiments/hinet_naked.pth"
    net.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))

    net.eval()

    print(net)
    print("[INFO] HINet naked model loaded.")

    with torch.no_grad():
        while True:
            ret, frame = video.read()
            if not ret:
                break
            # frame = frame[:, 180:1200, :]
            frame = cv2.resize(frame, (640, 368))
            input_image = frame.copy()
            input_image = preprocess(input_image)
            # pred = net(input_image)
            # pred_image_cpu = postprocess(pred)
            # pred_image_cpu = cv2.resize(pred_image_cpu, (frame.shape[1],frame.shape[0]))
            # image = np.concatenate((frame, pred_image_cpu), axis=1)
            # cv2.imshow("img", image)
            # if cv2.waitKey(1) == 27: break
            sample_image = input_image
            break
    
    # enable fp16 conversion

    net = net.half()
    
    torch.onnx.export(net, sample_image, "../experiments/hinet.onnx", verbose=True, input_names=["input"], output_names=["output"], opset_version=11)

    print("[FINISHED] HINet onnx model exported.")

    


if __name__ == '__main__':
    main()
