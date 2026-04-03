import asyncio
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

from app.core.config import STEMS_OUTPUT_DIR, get_torch_device

MODEL_NAME = "htdemucs"

# Module-level cache so the model is only loaded once per process lifetime.
_model_cache: dict = {}


def _get_model():
    if MODEL_NAME not in _model_cache:
        model = get_model(MODEL_NAME)
        model.eval()
        _model_cache[MODEL_NAME] = model
    return _model_cache[MODEL_NAME]


def _run_separation(input_wav: Path, job_id: str) -> dict:
    """
    Blocking function that runs Demucs stem separation.
    Intended to be called via asyncio.run_in_executor so it doesn't
    block the FastAPI event loop.
    """
    device = get_torch_device()
    model = _get_model()
    model.to(device)

    # Load audio via soundfile directly — torchaudio 2.11+ changed load()
    # to require torchcodec, so we bypass it entirely for WAV files.
    data, sr = sf.read(str(input_wav), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # [channels, samples]

    # Resample to model's expected sample rate if needed
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)

    # Demucs requires stereo; duplicate channel if input is mono
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]  # take first two channels if somehow > stereo

    # Normalize — Demucs is sensitive to scale
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    wav = (wav - mean) / (std + 1e-8)

    # apply_model expects shape [batch, channels, samples]
    with torch.no_grad():
        sources = apply_model(model, wav[None].to(device), device=device)[0]

    # Denormalize
    sources = sources * (std + 1e-8) + mean

    # Extract drums stem
    drums_idx = model.sources.index("drums")
    drums_wav = sources[drums_idx].cpu()  # shape: [2, samples]

    output_path = STEMS_OUTPUT_DIR / f"{job_id}_drums.wav"
    sf.write(str(output_path), drums_wav.numpy().T, model.samplerate)

    return {
        "job_id": job_id,
        "drums_file": output_path.name,
        "drums_path": str(output_path),
        "sample_rate": model.samplerate,
        "size_bytes": output_path.stat().st_size,
        "device_used": str(device),
    }


async def separate_stems(input_wav: Path, job_id: str) -> dict:
    """
    Async wrapper: runs the blocking Demucs separation in a thread pool
    so FastAPI stays responsive during the (potentially long) inference.

    Raises FileNotFoundError if input_wav doesn't exist.
    Raises RuntimeError if separation fails.
    """
    if not input_wav.exists():
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, partial(_run_separation, input_wav, job_id)
        )
    except Exception as exc:
        raise RuntimeError(f"Demucs separation failed: {exc}") from exc

    return result
