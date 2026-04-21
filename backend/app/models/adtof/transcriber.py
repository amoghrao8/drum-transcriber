"""
ADTOF-pytorch wrapper for drum transcription.

Uses the Frame_RNN model from https://github.com/xavriley/ADTOF-pytorch
(PyTorch port of ADTOF, F1 ~88.5% on MDBDrums++).

Output: 5 drum classes mapped to GM MIDI pitches:
  35 = kick   38 = snare   47 = toms (all merged)
  42 = hi-hat (open+closed)   49 = cymbal (crash+ride)
"""
from __future__ import annotations

from importlib import import_module
import logging
from pathlib import Path
import sys
from typing import Dict, List

import torch

log = logging.getLogger(__name__)

# ADTOF pitch → our GM pitch remapping
ADTOF_TO_GM: Dict[int, int] = {
    35: 36,   # Bass Drum → GM Kick
    38: 38,   # Snare     → GM Snare (same)
    47: 45,   # Tom       → GM Low Tom
    42: 42,   # Hi-hat    → GM Closed HH (same)
    49: 49,   # Cymbal    → GM Crash (same)
}

_model = None
_adtof_module = None
_REPO_ROOT = Path(__file__).resolve().parents[4]
_BUNDLED_PACKAGE_DIRS = (
    _REPO_ROOT / "adtof_pytorch" / "src",
    _REPO_ROOT / "adtof_pytorch",
)


def _import_adtof_package():
    global _adtof_module
    if _adtof_module is not None:
        return _adtof_module

    try:
        _adtof_module = import_module("adtof_pytorch")
        return _adtof_module
    except ModuleNotFoundError as exc:
        if exc.name != "adtof_pytorch":
            raise

    for candidate in _BUNDLED_PACKAGE_DIRS:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    try:
        _adtof_module = import_module("adtof_pytorch")
        return _adtof_module
    except ModuleNotFoundError as exc:
        if exc.name != "adtof_pytorch":
            raise
        search_paths = ", ".join(str(path) for path in _BUNDLED_PACKAGE_DIRS)
        raise RuntimeError(
            "ADTOF-pytorch is not available. Install backend requirements so pip can fetch "
            f"the package, or populate the bundled checkout at one of: {search_paths}"
        ) from exc


def _get_model():
    global _model
    if _model is None:
        adtof = _import_adtof_package()
        log.info("Loading ADTOF Frame_RNN model…")
        n_bins = adtof.calculate_n_bins()
        m = adtof.create_frame_rnn_model(n_bins)
        weights_path = adtof.get_default_weights_path()
        m = adtof.load_pytorch_weights(m, str(weights_path), strict=False)
        m.eval()
        _model = m
        log.info("ADTOF model loaded.")
    return _model


def transcribe_stem(wav_path: str) -> Dict[int, List[float]]:
    """
    Run ADTOF on a drum-stem WAV and return onset times per GM pitch.

    Returns
    -------
    dict mapping GM MIDI pitch → list of onset times in seconds
    e.g. {36: [0.39, 0.88, ...], 38: [1.01, 2.22, ...], ...}
    """
    adtof = _import_adtof_package()

    model = _get_model()
    x = adtof.load_audio_for_model(wav_path)

    with torch.no_grad():
        pred = model(x).cpu().numpy()

    picker = adtof.PeakPicker(thresholds=adtof.FRAME_RNN_THRESHOLDS, fps=100)
    raw = picker.pick(pred, labels=adtof.LABELS_5)[0]
    # raw = {35: [...], 38: [...], 47: [...], 42: [...], 49: [...]}

    # Remap ADTOF pitches to our GM constants
    return {ADTOF_TO_GM[adtof_pitch]: times
            for adtof_pitch, times in raw.items()
            if adtof_pitch in ADTOF_TO_GM}
