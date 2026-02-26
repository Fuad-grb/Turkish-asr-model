import torch
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from optimum.onnxruntime import ORTModelForCTC
from onnxruntime.quantization import quantize_dynamic, QuantType
import shutil

MODEL_DIR = "models/wav2vec2-tr-final"
ONNX_DIR = "models/onnx"
ONNX_QUANTIZED_DIR = "models/onnx-quantized"

os.makedirs(ONNX_DIR, exist_ok=True)
os.makedirs(ONNX_QUANTIZED_DIR, exist_ok=True)

print("[1/3] loading pytorch model...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

print("[2/3] exporting to onnx...")
ort_model = ORTModelForCTC.from_pretrained(MODEL_DIR, export=True)
ort_model.save_pretrained(ONNX_DIR)
processor.save_pretrained(ONNX_DIR)

onnx_file = os.path.join(ONNX_DIR, "model.onnx")
if not os.path.exists(onnx_file):
    for f in os.listdir(ONNX_DIR):
        if f.endswith(".onnx"):
            onnx_file = os.path.join(ONNX_DIR, f)
            break

print("[3/3] quantizing (dynamic int8)...")
quantized_file = os.path.join(ONNX_QUANTIZED_DIR, "model_quantized.onnx")
quantize_dynamic(
    model_input=onnx_file,
    model_output=quantized_file,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"]  # пропускаем Conv
)

for f in os.listdir(ONNX_DIR):
    if not f.endswith(".onnx") and not f.endswith(".onnx_data"):
        src = os.path.join(ONNX_DIR, f)
        dst = os.path.join(ONNX_QUANTIZED_DIR, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

pt_size = os.path.getsize(os.path.join(MODEL_DIR, "model.safetensors")) / 1024 / 1024
onnx_size = os.path.getsize(onnx_file) / 1024 / 1024
quant_size = os.path.getsize(quantized_file) / 1024 / 1024

print(f"\n--- Model Sizes ---")
print(f"PyTorch:    {pt_size:.1f} MB")
print(f"ONNX:       {onnx_size:.1f} MB")
print(f"Quantized:  {quant_size:.1f} MB")
print(f"Compression: {(1 - quant_size/pt_size)*100:.1f}%")
