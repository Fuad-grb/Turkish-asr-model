import numpy as np
import time
import librosa
import onnxruntime as ort
from transformers import Wav2Vec2Processor

QUANTIZED_MODEL = "models/onnx-quantized/model_quantized.onnx"
PROCESSOR_DIR = "models/onnx-quantized"

class ASRModel:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_DIR)
        self.session = ort.InferenceSession(
            QUANTIZED_MODEL,
            providers=["CPUExecutionProvider"]
        )
    
    def transcribe(self, audio_bytes: bytes, sr: int = None) -> dict:
        # audio from bites
        audio, orig_sr = librosa.load(
            __import__('io').BytesIO(audio_bytes),
            sr=16000, mono=True
        )
        
        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="np"
        )
        
        t0 = time.perf_counter()
        logits = self.session.run(None, {"input_values": inputs.input_values})[0]
        inference_time = time.perf_counter() - t0
        
        pred_ids = np.argmax(logits, axis=-1)
        text = self.processor.batch_decode(pred_ids)[0]
        
        return {
            "text": text.strip(),
            "inference_time": round(inference_time, 4)
        }
