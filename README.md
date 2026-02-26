# Turkish ASR Pipeline

Türk dili üçün Speech-to-Text sistemi. wav2vec2-large-xlsr-53 modeli Khan Academy Turkish dataseti üzərində fine-tune olunub, ONNX ilə optimallaşdırılıb və FastAPI + Docker ilə deploy olunub.

**Seçim B (Engineering Centric)** yanaşması — modelin WER-i ideal deyil (~34%), amma bütün sistem konteynerləşdirilib, optimallaşdırılıb və deploy-a hazırdır. Training asanlıqla daha çox data/epoch ilə yenidən başladıla bilər.

## Necə işə salmaq

**Docker:**
```bash
docker-compose up --build
# API http://localhost:8000 ünvanında
```

**Lokal:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/export_onnx.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Test:**
```bash
curl -X POST http://localhost:8000/transcribe -F "file=@audio.wav"
# => {"text": "merhaba dünya", "inference_time": 0.083}
```

Swagger UI: `http://localhost:8000/docs`

## Benchmark: PyTorch vs ONNX

CPU üzərində, 5 saniyəlik audio, 10 test ortalaması.

| Model | Ölçü | Latency | Sürətlənmə |
|-------|------|---------|------------|
| PyTorch (orijinal) | 1203.5 MB | 197.8 ms | 1x |
| ONNX | 1203.8 MB | 121.8 ms | 1.62x |
| ONNX + INT8 quantization | 338.6 MB | 83.6 ms | **2.37x** |

Quantization ilə modelin ölçüsü 72% kiçilib və sürət 2.37 dəfə artıb. Dynamic quantization MatMul/Gemm əməliyyatlarına tətbiq edilib (Conv layerləri ONNX Runtime-da weight normalization probleminə görə skip olunub).

## Training nəticələri

Google Colab-da (T4 GPU) icra olunub, `Turkish_asr_training.ipynb`-ə baxın.

- Baza model: `facebook/wav2vec2-large-xlsr-53`
- Dataset: `ysdede/khanacademy-turkish`-in 15%-i (~3.3k nümunə)
- 5 epoch, lr=3e-4, batch=4, gradient accumulation=4
- Feature encoder dondurulub, yalnız transformer + LM head öyrədilib
- TensorBoard ilə izlənib

WER 5 epoch ərzində 0.99-dan 0.34-ə düşüb. İdeal deyil, amma pipeline-ın işlədiyini sübut edir. Tam dataset ilə nəticə xeyli yaxşılaşar.

## Load test (Locust)

5 paralel istifadəçi, 30 saniyə:
- 61 sorğu, **0 xəta**
- Transcribe median: 71ms (warmup-dan sonra)
- İlk sorğular model yüklənməsinə görə yavaş (~7.5s)
- Health check: 3ms median

```bash
locust -f tests/locustfile.py --host http://localhost:8000 --headless -u 5 -r 2 -t 30s
```

## Layihə strukturu

```
scripts/
  export_onnx.py     - ONNX eksport + quantization
  benchmark.py       - sürət/ölçü müqayisəsi
app/
  main.py            - FastAPI endpoint-lər
  model.py           - ONNX inference logic
tests/
  locustfile.py      - yük testi
Turkish_asr_training.ipynb  - Colab training notebook
Dockerfile + docker-compose.yml
```

## Daha çox vaxt olsaydı nə edərdim

- Tam dataset üzərində training (daha aşağı WER)
- Static quantization (calibration dataset lazımdır)
- Multi-stage Docker build (daha kiçik image)
- GitHub Actions ilə CI/CD
- Daha yaxşı logging və error handling
- Audio preprocessing (səs-küy azaltma, VAD)
