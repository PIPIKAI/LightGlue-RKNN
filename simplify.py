import onnx
from onnxsim import simplify

model_path = "/root/codes/rknn_model_export/LightGlue-ONNX/output/onnx/superpoint_lightglue_end2end"
model = onnx.load(model_path + ".onnx")
model_simp, check = simplify(model=model)
assert check, "Simplified ONNX model could not be validated"
s_path = f"{model_path}_simplified.onnx"
onnx.save(model_simp, s_path)