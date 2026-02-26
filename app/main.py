from fastapi import FastAPI, UploadFile, File, HTTPException
from app.model import ASRModel

app = FastAPI(title="Turkish ASR API", version="1.0")
asr = ASRModel()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
        raise HTTPException(400, "supported formats: wav, mp3, flac, ogg")
    
    audio_bytes = await file.read()
    result = asr.transcribe(audio_bytes)
    return result
