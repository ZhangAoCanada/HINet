from os import pardir
from plistlib import InvalidFileException
import cv2
import numpy as np
import onnx
import onnxoptimizer
import onnxruntime as ort

from torchvision.transforms import Compose, ToTensor, Normalize

from skimage import img_as_ubyte

import time


def preprocessImage(input_img):
    # Resizing image in the multiple of 16"
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (640, 480), interpolation=cv2.INTER_AREA)
    input_img = input_img.astype(np.float32) / 255.

    input_img = np.expand_dims(input_img, axis=0)
    return input_img


model = onnx.load("../experiments/DeRain_512/models/hinet.onnx")
onnx.checker.check_model(model)

video_path = "/home/ao/tmp/clip_videos/h97cam_water_video.mp4"
cap = cv2.VideoCapture(video_path)

ort_session = ort.InferenceSession("../experiments/DeRain_512/models/hinet.onnx")

total_inference_time = 0
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    input_img = preprocessImage(frame)
    start = time.time()
    outputs = ort_session.run(
        None,
        {"input": input_img},
    )
    pred = outputs[0][0]
    total_inference_time += time.time() - start
    count += 1
    print("[INFO] average inference time: ", total_inference_time / count)
    # pred = pred * 255.0
    # pred = pred.astype(np.uint8)
    pred = img_as_ubyte(pred)
    pred = cv2.resize(pred, (frame.shape[1], frame.shape[0]))

    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    img_show = np.hstack((frame, pred))
    img_show = cv2.resize(img_show, None, fx=0.5, fy=0.5)
    cv2.imshow("pred", img_show)
    if cv2.waitKey(1) == 27:
        break
