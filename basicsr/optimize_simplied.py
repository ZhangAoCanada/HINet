# import onnx
# import onnxoptimizer
# import onnxruntime as ort
# model = onnx.load("./ckpt/transweather_sim.onnx")
# model_opt = onnxoptimizer.optimize(model)
# onnx.save(model_opt, "./ckpt/transweather.onnx")
# print("[INFO] finished.")

import onnx
from onnxsim import simplify
import onnxoptimizer
from onnxmltools.utils import float16_converter

onnx_model = onnx.load("../experiments/hinet.onnx")
# onnx_model = float16_converter.convert_float_to_float16(onnx_model,keep_io_types=True)
# model_simp = onnx_model

model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"

new_model = onnxoptimizer.optimize(model_simp)
onnx.save(new_model, "../experiments/hinet_simoptimized.onnx")
