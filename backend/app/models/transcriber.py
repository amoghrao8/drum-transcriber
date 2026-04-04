"""
Drum classifier — CNN architecture + spectral heuristic fallback.

If  backend/app/models/drum_classifier_weights.pth  exists it will be loaded
and the CNN path used.  Otherwise SpectralDrumClassifier provides a
high-accuracy heuristic based on FFT band-energy + decay analysis.

Classes (index → GM note):
  0  kick          → 36
  1  snare         → 38
  2  hihat_closed  → 42
  3  hihat_open    → 46
  4  ride          → 51
  5  crash         → 49
  6  china/trash   → 52
  7  clap/stack    → 39
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class registry
# ---------------------------------------------------------------------------

NUM_CLASSES  = 8
CLASSES      = ["kick", "snare", "hihat_closed", "hihat_open",
                "ride", "crash", "china", "clap"]
CLASS_NOTES  = [36, 38, 42, 46, 51, 49, 52, 39]

# ---------------------------------------------------------------------------
# Mel-spectrogram parameters  (torchaudio — used by CNN and as input feature)
# ---------------------------------------------------------------------------

N_MELS          = 64
N_FFT           = 512       # shorter window → better time resolution (was 1024)
HOP_LEN         = 441       # 10 ms at 44100 Hz (was 128 ≈ 2.9 ms)
WINDOW_SR       = 44100
WINDOW_MS       = 40        # 40 ms centered window (was 100 ms)
WINDOW_SAMPLES  = int(WINDOW_SR * WINDOW_MS / 1000)   # 1764 samples
WINDOW_HALF     = WINDOW_SAMPLES // 2                  # 882 samples = 20 ms

_WEIGHTS_PATH   = Path(__file__).parent / "drum_classifier_weights.pth"

# ---------------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------------

class DrumCNN(nn.Module):
    """
    Lightweight CNN for 40 ms drum hit windows.

    Input  : (B, 1, N_MELS, T)  — log-power Mel-spectrogram
    Output : (B, NUM_CLASSES)   — raw logits
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # type: ignore[override]
        return self.classifier(self.features(x))

# ---------------------------------------------------------------------------
# Mel-spectrogram extractor  (singleton transform, created once)
# ---------------------------------------------------------------------------

_mel_transform: T.MelSpectrogram | None = None


def _get_mel_transform() -> T.MelSpectrogram:
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = T.MelSpectrogram(
            sample_rate=WINDOW_SR,
            n_fft=N_FFT,
            hop_length=HOP_LEN,
            n_mels=N_MELS,
            f_min=20.0,
            f_max=20000.0,
        )
    return _mel_transform


def extract_mel(window: np.ndarray) -> torch.Tensor:
    """
    Convert a mono float32 numpy window to a (1, 1, N_MELS, T) log-power
    Mel-spectrogram tensor — ready to feed into DrumCNN.
    """
    if len(window) < WINDOW_SAMPLES:
        window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))
    else:
        window = window[:WINDOW_SAMPLES]

    t   = torch.from_numpy(window.copy()).unsqueeze(0)   # (1, samples)
    mel = _get_mel_transform()(t)                        # (1, N_MELS, T)
    mel = (mel + 1e-7).log()                             # log-power
    return mel.unsqueeze(0)                              # (1, 1, N_MELS, T)

# ---------------------------------------------------------------------------
# Spectral features  (FFT-based, fast)
# ---------------------------------------------------------------------------

class _Features(NamedTuple):
    e_sub:    float   # 20–150 Hz   — kick sub-bass
    e_bass:   float   # 150–400 Hz  — kick body
    e_lo_mid: float   # 400–1500 Hz — snare body
    e_mid:    float   # 1500–4000 Hz — snare highs / clap fundamental
    e_hi_mid: float   # 4000–8000 Hz — cymbal lower range
    e_hi:     float   # 8000–14000 Hz — hi-hat / ride bell
    e_vhi:    float   # 14000+ Hz   — china / crash top-end
    centroid: float   # spectral centroid (Hz)
    flatness: float   # spectral flatness (0 = tonal, 1 = noisy)
    decay:    float   # energy[20ms:40ms] / energy[0ms:20ms]  (post-onset window, 0ms = onset)
    rms:      float   # total window RMS


def compute_features(window: np.ndarray, sr: int = WINDOW_SR) -> _Features:
    """Compute spectral features from a raw mono float32 window."""
    n     = max(len(window), 2048)
    spec  = np.abs(np.fft.rfft(window, n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(spec[mask] ** 2)) + 1e-12

    total    = band(20, 22050)
    e_sub    = band(20,    150) / total
    e_bass   = band(150,   400) / total
    e_lo_mid = band(400,  1500) / total
    e_mid    = band(1500,  4000) / total
    e_hi_mid = band(4000,  8000) / total
    e_hi     = band(8000,  14000) / total
    e_vhi    = band(14000, 22050) / total

    # Spectral centroid
    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-8))

    # Spectral flatness  (geometric mean / arithmetic mean of power)
    power      = spec ** 2 + 1e-12
    geo_mean   = float(np.exp(np.mean(np.log(power))))
    arith_mean = float(np.mean(power))
    flatness   = geo_mean / arith_mean

    # Energy decay: second half of window vs first half
    mid      = max(len(window) // 2, 1)
    e_first  = float(np.mean(window[:mid] ** 2)) + 1e-12
    e_second = float(np.mean(window[mid:] ** 2)) + 1e-12
    decay    = e_second / e_first

    rms = float(np.sqrt(np.mean(window ** 2)))

    return _Features(
        e_sub=e_sub, e_bass=e_bass,
        e_lo_mid=e_lo_mid, e_mid=e_mid,
        e_hi_mid=e_hi_mid, e_hi=e_hi, e_vhi=e_vhi,
        centroid=centroid, flatness=flatness,
        decay=decay, rms=rms,
    )

# ---------------------------------------------------------------------------
# Spectral heuristic classifier
# ---------------------------------------------------------------------------

class SpectralDrumClassifier:
    """
    Rule-based drum classifier using FFT band-energy + spectral statistics.
    Provides three entry points:
      classify(window)         — unconstrained 8-class prediction
      classify_cymbal(window)  — constrained to cymbal classes (2–6)
      classify_snare(window)   — discriminates snare (1) vs clap (7)
    """

    # ── Full 8-class prediction ──────────────────────────────────────────────

    def classify(self, window: np.ndarray, sr: int = WINDOW_SR) -> int:
        """Return class index 0-7."""
        return self._rules(compute_features(window, sr))

    # ── Cymbal sub-classifier  (constrained to indices 2–6) ─────────────────

    def classify_cymbal(self, window: np.ndarray, sr: int = WINDOW_SR) -> int:
        """
        Classify a window known to contain a cymbal hit.
        Returns one of: 2=hihat_closed  3=hihat_open  4=ride  5=crash  6=china.
        """
        return self._rules_cymbal(compute_features(window, sr))

    def classify_cymbal_hybrid(
        self,
        window_band: np.ndarray,      # 5kHz+ filtered, 40ms — for spectral features
        window_full: np.ndarray,      # full mix, 40ms — for crash mid-energy
        window_band_long: np.ndarray, # 5kHz+ filtered, 150ms — for open/closed decay
        sr: int = WINDOW_SR,
    ) -> int:
        """
        Hybrid classifier:
          - 40ms band window  → spectral features (centroid, flatness, bands)
          - 40ms full window  → crash detection (needs mid-band plate resonance)
          - 150ms band window → open/closed decay (40ms is too short; closed HH
            ring at 10kHz+ lasts ~80ms, open HH lasts 200-400ms)

        Open/closed decay threshold uses energy[75ms:150ms] / energy[0ms:75ms].
        Closed HH: ring mostly gone by 75ms → decay ≈ 0.05-0.25
        Open HH:   ring still strong at 75ms → decay ≈ 0.30-0.75
        """
        f_band = compute_features(window_band, sr)
        f_full = compute_features(window_full, sr)

        # ── Long-window decay for open/closed distinction ─────────────────────
        mid_long = max(len(window_band_long) // 2, 1)
        e_first  = float(np.mean(window_band_long[:mid_long] ** 2)) + 1e-12
        e_second = float(np.mean(window_band_long[mid_long:] ** 2)) + 1e-12
        long_decay = e_second / e_first

        # ── Crash: full-signal mid+high simultaneously + sustained long decay ─
        mid_full = f_full.e_lo_mid + f_full.e_mid
        hi_full  = f_full.e_hi + f_full.e_vhi
        if mid_full > 0.12 and hi_full > 0.20 and long_decay > 0.30:
            return 5  # crash

        # ── China / Trash: very-high-freq + noisy in band signal ──────────────
        if f_band.e_vhi > 0.18 and f_band.flatness > 0.65:
            return 6  # china

        # ── Ride: extremely tonal bell (flatness < 0.12 — near-pure harmonic series)
        # True ride bell has flatness 0.05-0.12 in the 5kHz+ filtered signal.
        # Any hi-hat, even with cymbal resonance, has flatness > 0.15.
        if f_band.flatness < 0.12 and 6500 < f_band.centroid < 8500 and long_decay < 0.25:
            return 4  # ride

        # ── Open hi-hat: ring still present at 75-150ms ───────────────────────
        if long_decay > 0.25:
            return 3  # hihat_open

        return 2  # hihat_closed

    # ── Snare / clap discriminator ───────────────────────────────────────────

    def classify_snare(self, window: np.ndarray, sr: int = WINDOW_SR) -> int:
        """
        Returns 1 (snare) or 7 (clap/stack).
        Clap stacks: no low-end, no snare wire rattle (e_lo_mid), extremely fast decay.
        Snare: wire rattle always produces e_lo_mid ~0.15-0.35, even on ghost hits.
        """
        f = compute_features(window, sr)
        if (f.e_sub + f.e_bass) < 0.08 and f.e_lo_mid < 0.12 and f.decay < 0.15:
            return 7  # clap/stack
        return 1       # snare

    # ── Decision rules ───────────────────────────────────────────────────────

    @staticmethod
    def _rules(f: _Features) -> int:
        low_ratio    = f.e_sub + f.e_bass
        snare_ratio  = f.e_lo_mid + f.e_mid
        cymbal_ratio = f.e_hi_mid + f.e_hi + f.e_vhi

        # ── Kick: dominant low-end, centroid below 700 Hz ────────────────────
        if low_ratio > 0.30 and f.centroid < 700:
            return 0

        # ── Cymbal family ─────────────────────────────────────────────────────
        if cymbal_ratio > 0.45:
            return SpectralDrumClassifier._rules_cymbal(f)

        # ── Snare / clap ──────────────────────────────────────────────────────
        if snare_ratio > 0.30:
            if (f.e_sub + f.e_bass) < 0.08 and f.e_lo_mid < 0.12 and f.decay < 0.15:
                return 7   # clap
            return 1        # snare

        # ── Weak cymbal (e.g. quiet hi-hat) ──────────────────────────────────
        if cymbal_ratio > 0.25:
            return 2   # hihat_closed

        # ── Fallback ──────────────────────────────────────────────────────────
        if snare_ratio > 0.15:
            return 1
        return 0

    @staticmethod
    def _rules_cymbal(f: _Features) -> int:
        """
        Cymbal sub-classification.

        Rule ordering: crash first (requires mid+hi simultaneously), then china,
        then ride (strict flatness gate excludes hi-hats), then open/closed hi-hat.

        Key thresholds:
          Crash: mid_energy > 0.12 — plate resonance from a crash cymbal generates
                 real 400-4000 Hz energy; hi-hats have only bleed (~0.05-0.09).
          Ride:  flatness < 0.40 — ride bell is tonal (0.20-0.38); hi-hats are
                 metallic noise (0.55-0.75). Previous threshold 0.60 overlapped.
          Ride:  centroid 6000-9500 Hz — ride bell ring peak; hi-hats peak at
                 10000-13000 Hz. Previous 5000-11000 Hz range captured both.
        """
        mid_energy = f.e_lo_mid + f.e_mid    # 400-4000 Hz plate resonance
        hi_energy  = f.e_hi + f.e_vhi        # 8000 Hz+ shimmer

        # ── Crash: wide bandwidth + sustained ring in 20-40ms range ──────────
        # Crash has energy BOTH in mid (plate resonance) AND high (shimmer),
        # and still rings clearly at 20-40ms. Hi-hats have very little mid energy.
        if mid_energy > 0.12 and hi_energy > 0.20 and f.decay > 0.25:
            return 5  # crash

        # ── China / Trash Stack: extreme very-high-freq + maximum noise ───────
        if f.e_vhi > 0.18 and f.flatness > 0.65:
            return 6  # china

        # ── Ride: tonal bell in tight centroid band, NOT sustained like open HH ─
        # Ride bell: flatness 0.15-0.32, centroid 6500-8500 Hz, decay 0.20-0.45
        # Open HH:   flatness 0.50-0.75, centroid 9000-13000 Hz, decay > 0.45
        if f.flatness < 0.32 and 6500 < f.centroid < 8500 and f.decay < 0.50:
            return 4  # ride

        # ── Open hi-hat: ring still clearly present at 20-40ms ───────────────
        # Window is 0-40ms from onset. Decay = energy[20:40ms] / energy[0:20ms].
        # Closed HH: energy drops fast, decay 0.10-0.35.
        # Open HH: still ringing at 20ms, decay 0.40-0.75.
        if f.decay > 0.40:
            return 3  # hihat_open

        return 2  # hihat_closed — fast decay, high flatness, high centroid

# ---------------------------------------------------------------------------
# Factory — try CNN weights, fall back to heuristic
# ---------------------------------------------------------------------------

def load_classifier() -> tuple[DrumCNN | None, SpectralDrumClassifier]:
    """
    Return (cnn_model_or_None, heuristic).
    If drum_classifier_weights.pth exists in this directory, it is loaded
    into a DrumCNN and returned as the primary classifier.
    Otherwise only the SpectralDrumClassifier is returned.
    """
    heuristic = SpectralDrumClassifier()

    if not _WEIGHTS_PATH.exists():
        log.info("No weights at %s — using SpectralDrumClassifier", _WEIGHTS_PATH.name)
        return None, heuristic

    try:
        cnn = DrumCNN()
        cnn.load_state_dict(
            torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        )
        cnn.eval()
        log.info("Loaded DrumCNN weights from %s", _WEIGHTS_PATH.name)
        return cnn, heuristic
    except Exception as exc:
        log.warning("Weight load failed (%s) — using SpectralDrumClassifier", exc)
        return None, heuristic
