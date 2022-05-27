""" Dynamic Quantization """
# import onnx
# from onnxruntime.quantization import quantize, QuantizationMode

# model = onnx.load('./ckpt/transweather.onnx')
# quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps)
# onnx.save(quantized_model, './ckpt/transweather.quant.onnx')

# model = onnx.load("./ckpt/transweather.quant.onnx")
# onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))


""" Static Quantization """
import os
import numpy as np
import cv2
import onnx 
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.
    image = np.expand_dims(image, axis=0)
    # image = torch.from_numpy(image.astype(np.float32) / 255.0)
    return image


def postprocess(pred):
    pred = pred[0].float().detach().cpu().numpy()
    pred = (pred * 255.).round()
    pred = pred.astype(np.uint8)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    return pred


def preprocess_func(images_folder, size_limit=0):
    image_names = os.listdir(images_folder)
    print("[INFO] num of images: ", len(image_names))
    print("[INFO] image name sample: ", image_names[0])
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        # image_filepath = images_folder + '/' + image_name
        image_filepath = os.path.join(images_folder, image_name)
        image_data = preprocess_image(image_filepath)
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


class TransQuantDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, size_limit=10)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'input': nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)




if __name__ == "__main__":
    # calibration_image_folder = "/mnt/d/DATASET/DATA_2070/test/rain_L"
    calibration_image_folder = "/content/drive/MyDrive/DERAIN/DATA_20220325/test/rain_L"
    dr = TransQuantDataReader(calibration_image_folder)

    quantize_static("../experiments/DeRain_512/models/hinet.onnx", "../experiments/DeRain_512/models/hinet.quant.onnx", dr)
    # quantize_static("./ckpt/transweather.onnx", "./ckpt/transweather.quant.onnx", dr, extra_options={'ActivationSymmetric ': True, 'WeightSymmetric': True})

    print('ONNX full precision model size (MB):', os.path.getsize("../experiments/DeRain_512/models/hinet.onnx")/(1024*1024))
    print('ONNX quantized model size (MB):', os.path.getsize("../experiments/DeRain_512/models/hinet.quant.onnx")/(1024*1024))
