from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
_WINDOW_MS          = 50          # analysis window per onset (milliseconds)
_GHOST_THRESHOLD_DB = 15.0        # snare hits this many dB below peak → ghost
_HOP_LENGTH         = 512         # librosa onset hop (samples)

# Frequency band boundaries (Hz)
_KICK_MAX_HZ   = 100
_SNARE_MIN_HZ  = 200
_SNARE_MAX_HZ  = 3_000
_HAT_MIN_HZ    = 5_000


class DrumAnalyzer:
    """
    FFT-based drum transcriber operating on a pre-isolated drums stem.

    All per-onset computation (FFT, band power, RMS, ghost detection) is
    batched into vectorized NumPy operations so the hot path contains no
    Python-level loops over individual hits.
    """

    def __init__(self, sr: int = 44100) -> None:
        self.sr = sr
        self.window_samples = int(sr * _WINDOW_MS / 1_000)  # 2205 @ 44100 Hz

        # Pre-compute frequency axis and band masks once at init time.
        # Shape: (window_samples // 2 + 1,)
        self._freqs = np.fft.rfftfreq(self.window_samples, d=1.0 / sr)
        self._kick_mask  = self._freqs < _KICK_MAX_HZ
        self._snare_mask = (self._freqs >= _SNARE_MIN_HZ) & (self._freqs <= _SNARE_MAX_HZ)
        self._hat_mask   = self._freqs > _HAT_MIN_HZ

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, file_path: str | Path) -> list[dict]:
        """
        Analyse a drums-stem WAV and return a list of timestamped hit events.

        Returns
        -------
        list of dicts with keys:
            time      – onset time in seconds (float)
            type      – 'kick' | 'snare' | 'snare_ghost' | 'hat'
            velocity  – normalised RMS energy in [0, 1] (float)
        """
        y, sr = librosa.load(str(file_path), sr=self.sr, mono=True)

        # --- 1. Onset detection with backtracking -----------------------
        # backtrack=True walks back to the nearest local energy minimum so
        # onset times sit at the physical attack transient, not the envelope
        # peak.
        onset_samples: np.ndarray = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            backtrack=True,
            units="samples",
            hop_length=_HOP_LENGTH,
        )

        if onset_samples.size == 0:
            return []

        # --- 2. Extract analysis windows --------------------------------
        # Produces (N, window_samples) float32 matrix.
        # This loop is over onsets (unavoidable for arbitrary positions) but
        # all downstream computation is fully vectorized over the batch.
        windows = self._extract_windows(y, onset_samples)

        # --- 3. Vectorized FFT classification ---------------------------
        drum_types, _band_powers = self._classify_windows(windows)

        # --- 4. Vectorized RMS → velocity [0, 1] ------------------------
        # np.mean over axis=1 operates on the full (N, W) matrix at once.
        rms = np.sqrt(np.mean(windows ** 2, axis=1))           # (N,)
        velocity = rms / (rms.max() + 1e-8)                    # normalise

        # --- 5. Ghost-note detection on snare subset --------------------
        is_ghost = self._detect_ghosts(drum_types, rms)

        # --- 6. Build output --------------------------------------------
        onset_times = onset_samples / sr
        events: list[dict] = []
        for i, t in enumerate(onset_times):
            hit_type = "snare_ghost" if is_ghost[i] else drum_types[i]
            events.append({
                "time":     round(float(t), 4),
                "type":     hit_type,
                "velocity": round(float(velocity[i]), 4),
            })

        return events

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_windows(
        self, y: np.ndarray, onset_samples: np.ndarray
    ) -> np.ndarray:
        """
        Extract a fixed-size window starting at each onset sample.
        Windows that reach past the end of the signal are zero-padded.

        Returns
        -------
        windows : np.ndarray, shape (N, window_samples), float32
        """
        n = len(onset_samples)
        windows = np.zeros((n, self.window_samples), dtype=np.float32)
        for i, s in enumerate(onset_samples):
            end = min(int(s) + self.window_samples, len(y))
            windows[i, : end - int(s)] = y[int(s) : end]
        return windows

    def _classify_windows(
        self, windows: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        """
        Classify each onset window by its dominant frequency band.

        Strategy
        --------
        1. Compute rfft for all windows simultaneously → (N, F) complex matrix.
        2. Compute power spectrum |rfft|² → (N, F).
        3. Sum power inside each band using pre-computed boolean masks.
        4. argmax over the three band totals selects the instrument label.

        Returns
        -------
        drum_types  : list[str] length N
        band_powers : np.ndarray shape (N, 3)  [kick, snare, hat]
        """
        # Vectorized FFT over entire batch — shape: (N, window_samples//2 + 1)
        spectra = np.fft.rfft(windows, axis=1)
        power   = np.abs(spectra) ** 2                          # (N, F)

        # Band energy sums — boolean mask indexing keeps this loop-free
        kick_power  = power[:, self._kick_mask].sum(axis=1)     # (N,)
        snare_power = power[:, self._snare_mask].sum(axis=1)    # (N,)
        hat_power   = power[:, self._hat_mask].sum(axis=1)      # (N,)

        # (N, 3) → argmax gives index of dominant band per onset
        band_powers = np.stack([kick_power, snare_power, hat_power], axis=1)
        dominant    = np.argmax(band_powers, axis=1)             # (N,)

        _labels    = ["kick", "snare", "hat"]
        drum_types = [_labels[d] for d in dominant]

        return drum_types, band_powers

    def _detect_ghosts(
        self, drum_types: list[str], rms: np.ndarray
    ) -> np.ndarray:
        """
        Flag snare hits whose RMS energy is ≥ GHOST_THRESHOLD_DB below the
        track's average snare peak (defined as the mean of the top quartile
        of snare hits by energy).

        Returns
        -------
        is_ghost : np.ndarray, shape (N,), dtype bool
        """
        is_ghost = np.zeros(len(drum_types), dtype=bool)

        snare_idx = np.array(
            [i for i, t in enumerate(drum_types) if t == "snare"], dtype=int
        )
        if snare_idx.size < 2:
            return is_ghost

        snare_rms = rms[snare_idx]                              # subset
        snare_db  = 20.0 * np.log10(snare_rms + 1e-8)          # (M,)

        # Reference peak: mean energy of the loudest 25 % of snare hits
        peak_db = snare_db[snare_db >= np.percentile(snare_db, 75)].mean()

        ghost_mask               = snare_db < (peak_db - _GHOST_THRESHOLD_DB)
        is_ghost[snare_idx[ghost_mask]] = True

        return is_ghost
