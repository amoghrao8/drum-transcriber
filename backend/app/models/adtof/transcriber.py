"""
ADTOF-pytorch wrapper for drum transcription.

Uses the Frame_RNN model from https://github.com/xavriley/ADTOF-pytorch
(PyTorch port of ADTOF, F1 ~88.5% on MDBDrums++).

Output: 5 drum classes mapped to GM MIDI pitches:
  35 = kick   38 = snare   47 = toms (all merged)
  42 = hi-hat (open+closed)   49 = cymbal (crash+ride)
"""
from __future__ import annotations

import logging
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


def _get_model():
    global _model
    if _model is None:
        from adtof_pytorch import (
            create_frame_rnn_model,
            calculate_n_bins,
            load_pytorch_weights,
            get_default_weights_path,
        )
        log.info("Loading ADTOF Frame_RNN model…")
        n_bins = calculate_n_bins()
        m = create_frame_rnn_model(n_bins)
        weights_path = get_default_weights_path()
        m = load_pytorch_weights(m, str(weights_path), strict=False)
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
    from adtof_pytorch import (
        load_audio_for_model,
        PeakPicker,
        FRAME_RNN_THRESHOLDS,
        LABELS_5,
    )

    model = _get_model()
    x = load_audio_for_model(wav_path)

    with torch.no_grad():
        pred = model(x).cpu().numpy()

    picker = PeakPicker(thresholds=FRAME_RNN_THRESHOLDS, fps=100)
    raw = picker.pick(pred, labels=LABELS_5)[0]
    # raw = {35: [...], 38: [...], 47: [...], 42: [...], 49: [...]}

    # Remap ADTOF pitches to our GM constants
    return {ADTOF_TO_GM[adtof_pitch]: times
            for adtof_pitch, times in raw.items()
            if adtof_pitch in ADTOF_TO_GM}
