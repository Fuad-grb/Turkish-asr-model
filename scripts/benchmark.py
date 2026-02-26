import time
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import onnxruntime as ort

MODEL_DIR = "models/wav2vec2-tr-final"
ONNX_DIR = "models/onnx"
ONNX_QUANTIZED_DIR = "models/onnx-quantized"

processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

# generate 5 sec fake audio
dummy_audio = np.random.randn(16000 * 5).astype(np.float32)
inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="np")
input_values_np = inputs.input_values

N_RUNS = 10

# --- PyTorch ---
print("benchmarking pytorch...")
pt_model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
pt_model.eval()
input_values_pt = torch.tensor(input_values_np)

with torch.no_grad():
    # warmup
    for _ in range(3):
        pt_model(input_values_pt)
    
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        pt_model(input_values_pt)
        times.append(time.perf_counter() - t0)
pt_avg = np.mean(times) * 1000

# --- ONNX ---
print("benchmarking onnx...")
onnx_file = [f for f in __import__('os').listdir(ONNX_DIR) if f.endswith('.onnx')][0]
sess = ort.InferenceSession(f"{ONNX_DIR}/{onnx_file}", providers=["CPUExecutionProvider"])

for _ in range(3):
    sess.run(None, {"input_values": input_values_np})

times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    sess.run(None, {"input_values": input_values_np})
    times.append(time.perf_counter() - t0)
onnx_avg = np.mean(times) * 1000

# --- ONNX Quantized ---
print("benchmarking quantized onnx...")
sess_q = ort.InferenceSession(f"{ONNX_QUANTIZED_DIR}/model_quantized.onnx", providers=["CPUExecutionProvider"])

for _ in range(3):
    sess_q.run(None, {"input_values": input_values_np})

times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    sess_q.run(None, {"input_values": input_values_np})
    times.append(time.perf_counter() - t0)
quant_avg = np.mean(times) * 1000

print(f"\n{'='*45}")
print(f"{'Model':<20} {'Size (MB)':<12} {'Latency (ms)'}")
print(f"{'='*45}")

import os
pt_size = os.path.getsize(f"{MODEL_DIR}/model.safetensors") / 1024 / 1024
onnx_size = os.path.getsize(f"{ONNX_DIR}/{onnx_file}") / 1024 / 1024
quant_size = os.path.getsize(f"{ONNX_QUANTIZED_DIR}/model_quantized.onnx") / 1024 / 1024

print(f"{'PyTorch':<20} {pt_size:<12.1f} {pt_avg:.1f}")
print(f"{'ONNX':<20} {onnx_size:<12.1f} {onnx_avg:.1f}")
print(f"{'ONNX Quantized':<20} {quant_size:<12.1f} {quant_avg:.1f}")
print(f"{'='*45}")
print(f"ONNX speedup: {pt_avg/onnx_avg:.2f}x")
print(f"Quantized speedup: {pt_avg/quant_avg:.2f}x")
