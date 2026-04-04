"""
MDX23C-DrumSep wrapper using ZFTurbo's Music-Source-Separation-Training framework.

Two models:
  load_model()       — aufr33/jarredou 6-stem (kick/snare/toms/hh/ride/crash)
  load_5stem_model() — jarredou 5-stem (kick/snare/toms/hh/cymbals), higher SDR
Both share the same separate() inference function.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import soundfile as sf

# ---------------------------------------------------------------------------
# ZFTurbo framework path — cloned at drum-transcriber/zftturbo_mss/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent / "zftturbo_mss"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Default paths relative to drum-transcriber root
_MODELS_ROOT = Path(__file__).parent.parent.parent.parent.parent / "pretrained_mdx23c_models"

_DEFAULT_6STEM_CKPT    = _MODELS_ROOT / "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt"
_DEFAULT_6STEM_CONFIG  = _MODELS_ROOT / "aufr33-jarredou_DrumSep_config.yaml"
_DEFAULT_5STEM_CKPT    = _MODELS_ROOT / "drumsep_5stems_jarredou.ckpt"
_DEFAULT_5STEM_CONFIG  = _MODELS_ROOT / "drumsep_5stems_jarredou_config.yaml"

# Keep old names as aliases for backwards compat
_DEFAULT_CKPT   = _DEFAULT_6STEM_CKPT
_DEFAULT_CONFIG = _DEFAULT_6STEM_CONFIG


def _load(ckpt_path: str | Path, config_path: str | Path, device: str):
    """Shared loader for any MDX23C checkpoint."""
    from utils.settings import get_model_from_config  # ZFTurbo
    model, config = get_model_from_config("mdx23c", str(config_path))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "state"):
        if key in ckpt:
            ckpt = ckpt[key]
            break
    model.load_state_dict(ckpt)
    model.eval()
    model.to(device)
    return model, config, torch.device(device)


def load_model(
    ckpt_path:   str | Path = _DEFAULT_6STEM_CKPT,
    config_path: str | Path = _DEFAULT_6STEM_CONFIG,
    device:      str        = "cpu",
):
    """Load aufr33/jarredou 6-stem model (kick/snare/toms/hh/ride/crash)."""
    return _load(ckpt_path, config_path, device)


def load_5stem_model(
    ckpt_path:   str | Path = _DEFAULT_5STEM_CKPT,
    config_path: str | Path = _DEFAULT_5STEM_CONFIG,
    device:      str        = "cpu",
):
    """Load jarredou 5-stem model (kick/snare/toms/hh/cymbals) — higher SDR."""
    return _load(ckpt_path, config_path, device)


def separate(
    file_path: str | Path,
    model,
    config,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run MDX23C-DrumSep on a drum stem WAV.

    Parameters
    ----------
    file_path : path to the drum-stem WAV (any sr; resampled to 44100 if needed)
    model     : loaded MDX23C model (from load_model)
    config    : ZFTurbo config object (from load_model)
    device    : torch.device

    Returns
    -------
    dict mapping stem name → mono float32 numpy array at original sr
    """
    from utils.model_utils import demix  # ZFTurbo

    data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
    # MDX23C expects stereo (2, N) at 44100 Hz
    mix = data.T  # (channels, samples)

    if sr != config.audio.sample_rate:
        import torchaudio as _ta
        mix_t = torch.from_numpy(mix).float()
        mix_t = _ta.functional.resample(mix_t, sr, config.audio.sample_rate)
        mix = mix_t.numpy()

    # demix returns dict {instrument: ndarray (channels, samples)} for mdx23c
    with torch.no_grad():
        result = demix(config, model, mix, device, model_type="mdx23c", pbar=False)

    # Convert each stem to mono float32
    out: Dict[str, np.ndarray] = {}
    for stem_name, arr in result.items():
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        out[stem_name] = arr.astype(np.float32)

    return out
