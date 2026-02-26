# Turkish ASR Pipeline — Wav2Vec2 Fine-tuning & ONNX Optimization

Speech-to-Text system for Turkish language built on wav2vec2-large-xlsr-53 with ONNX optimization for production deployment.

## Benchmark Report

### Model Size

| Model          | Size (MB) | Compression |
|----------------|-----------|-------------|
| PyTorch        | 1203.5    | —           |
| ONNX           | 1203.8    | ~0%         |
| ONNX Quantized | 338.6     | **71.9%**   |

### Inference Latency (5s audio, CPU, 10 runs avg)

| Model          | Latency (ms) | Speedup |
|----------------|-------------|---------|
| PyTorch        | 197.8       | 1.00x   |
| ONNX           | 121.8       | 1.62x   |
| ONNX Quantized | 83.6        | **2.37x** |

### Training Results

- **Base model:** facebook/wav2vec2-large-xlsr-53
- **Dataset:** ysdede/khanacademy-turkish (15% subset, ~3.3k samples)
- **Epochs:** 5
- **Final WER:** 0.34 (34%)
- **Tracking:** TensorBoard

| Epoch | Train Loss | Val Loss | WER    |
|-------|-----------|----------|--------|
| 1     | 12.40     | 2.94     | 0.998  |
| 2     | 3.38      | 0.56     | 0.561  |
| 3     | 2.31      | 0.42     | 0.416  |
| 4     | 1.98      | 0.38     | 0.355  |
| 5     | 1.69      | 0.37     | 0.340  |

> WER is intentionally not optimized — this is an engineering-centric submission. The pipeline supports easy retraining with more data/epochs.

### Load Testing (Locust, 5 concurrent users, 30s)

| Metric                | Value     |
|-----------------------|-----------|
| Total requests        | 61        |
| Failure rate          | 0%        |
| Transcribe median     | 71 ms     |
| Transcribe p95        | 7500 ms*  |
| Health check median   | 3 ms      |

*p95 spike due to model cold-start on first requests; steady-state latency is 60-90ms.

## Quick Start

### Option 1: Docker (recommended)

```bash
docker-compose up --build
```

API will be available at `http://localhost:8000`

### Option 2: Local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# export model (requires trained model in models/wav2vec2-tr-final/)
python scripts/export_onnx.py

# run API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Transcribe audio

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav"
```

Response:
```json
{
  "text": "merhaba dünya",
  "inference_time": 0.083
}
```

### Health check

```bash
curl http://localhost:8000/health
```

### Swagger UI

Open `http://localhost:8000/docs` in browser.

## Training

Training was performed in Google Colab (T4 GPU). See `Turkish_asr_training.ipynb` for the full notebook.

To retrain with different parameters:
1. Open the notebook in Colab
2. Adjust subset size, epochs, learning rate in the training cells
3. Download the model and place in `models/wav2vec2-tr-final/`
4. Run `python scripts/export_onnx.py` to re-export

## Load Testing

```bash
# start API first, then:
locust -f tests/locustfile.py --host http://localhost:8000 --headless -u 5 -r 2 -t 30s
```

## Tech Stack

- **Model:** wav2vec2-large-xlsr-53 (fine-tuned)
- **Optimization:** ONNX Runtime + Dynamic INT8 Quantization
- **API:** FastAPI + Uvicorn
- **Container:** Docker (python:3.12-slim)
- **Load Testing:** Locust
- **Training Tracking:** TensorBoard
