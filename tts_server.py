import os
import sys
import io
import inspect
import tempfile

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from huggingface_hub import hf_hub_download

def safe_audio_load(filepath, **kwargs):
    data, sr = sf.read(filepath, dtype="float32")
    if data.ndim == 1:
        tensor = torch.from_numpy(data).unsqueeze(0).float()
    else:
        tensor = torch.from_numpy(data).t().float()
        tensor = tensor.mean(dim=0, keepdim=True)
    target_sr = 24000
    if sr != target_sr:
        tensor = F.resample(tensor, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
    return tensor, sr

torchaudio.load = safe_audio_load

from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process
from f5_tts.model import DiT

app = FastAPI(title="Atlas Audio Cortex")

F5_MODEL = None
VOCODER = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
REF_PREP_CACHE = {}

class TTSRequest(BaseModel):
    gen_text: str
    ref_audio_path: str
    ref_text: str
    speed: float = 1.0

def _prepare_clean_reference(ref_audio_path: str, ref_text: str) -> tuple[str, str]:
    """Adds 500ms trailing silence to the reference WAV and ensures ref_text ends with '. ' (no ffmpeg needed)."""
    mtime = os.path.getmtime(ref_audio_path)
    key = (ref_audio_path, mtime, ref_text.strip())
    if key in REF_PREP_CACHE and os.path.isfile(REF_PREP_CACHE[key][0]):
        return REF_PREP_CACHE[key]

    data, sr = sf.read(ref_audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 24000:
        t = torch.from_numpy(data).unsqueeze(0).float()
        data = F.resample(t, orig_freq=sr, new_freq=24000).squeeze(0).numpy()
        sr = 24000

    # Trim dead leading/trailing silence, then append 0.50s of clean zero-silence
    non_silent = np.where(np.abs(data) > 0.008)[0]
    if non_silent.size > 100:
        pad_head = int(0.05 * sr)
        pad_tail = int(0.08 * sr)
        s_idx = max(0, int(non_silent[0]) - pad_head)
        e_idx = min(len(data), int(non_silent[-1]) + pad_tail)
        data = data[s_idx:e_idx]

    # Normalize RMS around ~0.12 so F5-TTS mel conditioning stays in-distribution
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size > 0 else 0.0
    if rms > 1e-4:
        data = data * (0.12 / rms)
    peak = float(np.max(np.abs(data))) if data.size > 0 else 0.0
    if peak > 0.95:
        data = data * (0.95 / peak)

    silence_500ms = np.zeros(int(0.50 * sr), dtype=np.float32)
    padded = np.concatenate([data, silence_500ms], axis=0)

    stem = os.path.splitext(os.path.basename(ref_audio_path))[0]
    out_wav = os.path.join(tempfile.gettempdir(), f"f5_ref_padded_{stem}.wav")
    sf.write(out_wav, padded, sr, format="WAV", subtype="PCM_16")

    clean_ref_txt = ref_text.strip()
    if clean_ref_txt and clean_ref_txt[-1] not in ".!?,;":
        clean_ref_txt += "."
    clean_ref_txt += " "

    REF_PREP_CACHE[key] = (out_wav, clean_ref_txt)
    return out_wav, clean_ref_txt

@app.on_event("startup")
async def load_engine():
    global F5_MODEL, VOCODER
    print("[F5-TTS] Booting F5-TTS Zero-Shot Engine into VRAM on Port 1237...", flush=True)
    model_path = hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_Base/model_1200000.safetensors")
    try:
        vocab_path = hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_Base/vocab.txt")
    except Exception:
        vocab_path = ""

    # Match exact F5TTS_Base (v0) DiT architecture so newer f5-tts installs do not apply v1 defaults
    dit_params = inspect.signature(DiT.__init__).parameters
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    if "text_mask_padding" in dit_params:
        model_cfg["text_mask_padding"] = False
    if "pe_attn_head" in dit_params:
        model_cfg["pe_attn_head"] = 1

    print(f"[F5-TTS] DiT Config Locked: {model_cfg} | Vocab: {os.path.basename(vocab_path) or 'default'}", flush=True)

    F5_MODEL = load_model(
        model_cls=DiT,
        model_cfg=model_cfg,
        ckpt_path=model_path,
        vocab_file=vocab_path,
        mel_spec_type="vocos",
        device=DEVICE,
    )
    F5_MODEL = F5_MODEL.to(torch.float32)
    VOCODER = load_vocoder(is_local=False)
    print("[F5-TTS] Audio Cortex Online and standing by on http://127.0.0.1:1237", flush=True)

@app.get("/health")
async def health_check():
    if F5_MODEL is not None and VOCODER is not None:
        return {"status": "ok", "device": DEVICE}
    raise HTTPException(status_code=503, detail="Loading F5-TTS...")

@app.post("/v1/audio/generate")
async def generate_speech(request: TTSRequest):
    if not F5_MODEL or not VOCODER:
        raise HTTPException(status_code=503, detail="TTS Engine is not fully loaded.")
    if not os.path.exists(request.ref_audio_path):
        raise HTTPException(status_code=404, detail=f"Reference audio not found at {request.ref_audio_path}")

    gen_clean = " ".join(request.gen_text.strip().split())
    if gen_clean and gen_clean[-1] not in ".!?":
        gen_clean += "."

    print(f"[F5-TTS] Synthesizing on {DEVICE}: {gen_clean[:70]}", flush=True)
    try:
        prep_wav, prep_txt = _prepare_clean_reference(request.ref_audio_path, request.ref_text)
        audio_out, sample_rate, _ = infer_process(
            ref_audio=prep_wav,
            ref_text=prep_txt,
            gen_text=gen_clean,
            model_obj=F5_MODEL,
            vocoder=VOCODER,
            mel_spec_type="vocos",
            speed=request.speed,
            device=DEVICE,
        )

        audio_np = np.asarray(audio_out, dtype=np.float32).squeeze()
        peak = float(np.max(np.abs(audio_np))) if audio_np.size > 0 else 0.0
        if peak > 0.96:
            audio_np = audio_np * (0.92 / peak)

        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=1237)