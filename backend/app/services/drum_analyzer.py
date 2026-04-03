"""
Multi-Pass PyTorch drum transcriber — Klangio-inspired pipeline.

Pipeline
────────
Pass 1  Per-band onset detection
          • Kick band   50–200 Hz   → precise kick timing
          • Snare band  200–5 kHz   → snare / clap timing (kick-bleed suppressed)
          • Cymbal band 5 kHz+      → all cymbal timing

Pass 2  Feature extraction per onset
          • Extract 100 ms window from the FULL signal (not the filtered one)
            so the classifier has access to the complete spectral picture
          • Compute torchaudio Mel-spectrogram for CNN path
          • Compute FFT band-energy + spectral statistics for heuristic path

Pass 3  Instrument classification
          • Kick onsets     → always GM_KICK (spectral confirmation only)
          • Snare onsets    → SpectralDrumClassifier.classify_snare()
                              → GM_SNARE or GM_CLAP
          • Cymbal onsets   → SpectralDrumClassifier.classify_cymbal()
                              → GM_HIHAT_CLOSED / GM_HIHAT_OPEN /
                                 GM_RIDE / GM_CRASH / GM_CHINA
          • CNN path (if weights present): DrumCNN replaces heuristic

Pass 4  Post-processing
          • Velocity estimation from per-onset RMS
          • Ghost-note tagging  (snare < 30th-percentile velocity)
          • 16th-note quantisation + deduplication
          • GM MIDI export (mido)

General MIDI percussion (channel 9) note map
────────────────────────────────────────────
  36 Kick   38 Snare   39 Clap/Stack
  42 HH Closed   46 HH Open
  49 Crash   51 Ride   52 China/Trash
"""
from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import librosa
import mido
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt

from app.models.transcriber import (
    CLASS_NOTES,
    CLASSES,
    DrumCNN,
    SpectralDrumClassifier,
    compute_features,
    extract_mel,
    load_classifier,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GM drum note constants
# ---------------------------------------------------------------------------
GM_KICK          = 36
GM_SNARE         = 38
GM_CLAP          = 39
GM_HIHAT_CLOSED  = 42
GM_HIHAT_OPEN    = 46
GM_CRASH         = 49
GM_RIDE          = 51
GM_CHINA         = 52

GM_NAMES: dict[int, str] = {
    GM_KICK:         "kick",
    GM_SNARE:        "snare",
    GM_CLAP:         "clap",
    GM_HIHAT_CLOSED: "hihat_closed",
    GM_HIHAT_OPEN:   "hihat_open",
    GM_CRASH:        "crash",
    GM_RIDE:         "ride",
    GM_CHINA:        "china",
}

# Lazy-initialised classifier (loaded once at first transcription call)
_cnn:       DrumCNN | None              = None
_heuristic: SpectralDrumClassifier | None = None


def _get_classifier() -> tuple[DrumCNN | None, SpectralDrumClassifier]:
    global _cnn, _heuristic
    if _heuristic is None:
        _cnn, _heuristic = load_classifier()
    return _cnn, _heuristic


# ---------------------------------------------------------------------------
# Butterworth band / high / low-pass filter
# ---------------------------------------------------------------------------

def _butter_filter(y: np.ndarray, sr: int, ftype: str, freqs) -> np.ndarray:
    sos = butter(4, freqs, btype=ftype, fs=sr, output="sos")
    return sosfilt(sos, y).astype(np.float32)


# ---------------------------------------------------------------------------
# Onset detection helpers
# ---------------------------------------------------------------------------

def _onset_times(
    y: np.ndarray,
    sr: int,
    *,
    delta: float = 0.07,
    wait: int    = 4,
) -> np.ndarray:
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        backtrack=True,
        units="frames",
        pre_max=3, post_max=3,
        pre_avg=3, post_avg=5,
        delta=delta,
        wait=wait,
        hop_length=256,
    )
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)


def _rms_at(y: np.ndarray, sr: int, times: np.ndarray, window_s: float = 0.04) -> np.ndarray:
    """Peak RMS in a short window after each onset."""
    w   = int(sr * window_s)
    rms = np.zeros(len(times), dtype=np.float32)
    for i, t in enumerate(times):
        s = int(t * sr)
        chunk = y[s : s + w]
        if len(chunk):
            rms[i] = float(np.sqrt(np.mean(chunk ** 2)))
    return rms


def _rms_to_velocity(rms: np.ndarray, lo: int = 30, hi: int = 110) -> np.ndarray:
    if rms.max() < 1e-8:
        return np.full(len(rms), lo, dtype=np.int32)
    norm = rms / rms.max()
    return np.clip((norm * (hi - lo) + lo).round(), lo, hi).astype(np.int32)


# ---------------------------------------------------------------------------
# BPM & beat grid
# ---------------------------------------------------------------------------

def _detect_bpm(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    bpm = float(np.squeeze(tempo))
    return bpm, librosa.frames_to_time(beat_frames, sr=sr)


# ---------------------------------------------------------------------------
# Time signature inference  (autocorrelation of onset strength envelope)
# ---------------------------------------------------------------------------

def _infer_time_signature(
    y: np.ndarray,
    sr: int,
    bpm: float,
    beat_times: np.ndarray,
) -> tuple[int, int]:
    if len(beat_times) < 8:
        return 4, 4

    onset_env   = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    ac          = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
    sr_onset    = sr / 512
    beat_frames = round(60.0 / bpm * sr_onset)

    scores: dict[int, float] = {}
    for n in (3, 4, 5, 6, 7):
        frame = beat_frames * n
        if frame < len(ac):
            scores[n] = float(ac[int(frame)])

    if not scores:
        return 4, 4

    best = max(scores, key=scores.__getitem__)
    if best != 4 and scores.get(best, 0) > scores.get(4, 0) * 1.3:
        return best, 4
    return 4, 4


# ---------------------------------------------------------------------------
# Pass 3 helpers — classify a window extracted from the full signal
# ---------------------------------------------------------------------------

def _classify_cymbal(
    t: float,
    y_full: np.ndarray,
    sr: int,
    heuristic: SpectralDrumClassifier,
    cnn: DrumCNN | None,
) -> int:
    """
    Return GM note for a cymbal-band onset.
    Uses the full-signal window so the classifier can see crash mid-spread.
    """
    s      = int(t * sr)
    window = y_full[s : s + int(sr * 0.1)].astype(np.float32)

    if cnn is not None:
        with torch.no_grad():
            mel    = extract_mel(window)
            logits = cnn(mel)[0]
            # Restrict to cymbal classes (2–6)
            cymbal_logits = logits[[2, 3, 4, 5, 6]]
            cls_local     = int(cymbal_logits.argmax().item())
            cls_global    = [2, 3, 4, 5, 6][cls_local]
        return CLASS_NOTES[cls_global]

    cls = heuristic.classify_cymbal(window, sr)
    return CLASS_NOTES[cls]


def _classify_snare(
    t: float,
    y_full: np.ndarray,
    sr: int,
    heuristic: SpectralDrumClassifier,
    cnn: DrumCNN | None,
) -> int:
    """Return GM_SNARE or GM_CLAP for a snare-band onset."""
    s      = int(t * sr)
    window = y_full[s : s + int(sr * 0.1)].astype(np.float32)

    if cnn is not None:
        with torch.no_grad():
            mel    = extract_mel(window)
            logits = cnn(mel)[0]
            snare_logits = logits[[1, 7]]
            cls_local    = int(snare_logits.argmax().item())
            cls_global   = [1, 7][cls_local]
        return CLASS_NOTES[cls_global]

    cls = heuristic.classify_snare(window, sr)
    return CLASS_NOTES[cls]


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------

def _quantize(events: list[dict], bpm: float, beats_per_bar: int = 4) -> list[dict]:
    grid_dur = 60.0 / bpm / 4
    seen: set[tuple[float, int]] = set()
    out:  list[dict]             = []
    for ev in sorted(events, key=lambda e: e["time"]):
        idx    = round(ev["time"] / grid_dur)
        q_time = round(idx * grid_dur, 4)
        key    = (q_time, ev["note"])
        if key not in seen:
            seen.add(key)
            out.append({**ev, "time": q_time})
    return out


# ---------------------------------------------------------------------------
# Ghost-note tagging
# ---------------------------------------------------------------------------

def _tag_ghosts(events: list[dict]) -> list[dict]:
    snare_vels = [e["velocity"] for e in events if e["note"] == GM_SNARE]
    if len(snare_vels) < 4:
        return [{**e, "ghost": False} for e in events]

    threshold = float(np.percentile(snare_vels, 30))
    return [
        {**ev, "ghost": ev["note"] == GM_SNARE and ev["velocity"] < threshold}
        for ev in events
    ]


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------

def _build_midi(
    events: list[dict],
    bpm: float,
    beats_per_bar: int,
    beat_unit: int,
) -> mido.MidiFile:
    mid   = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo",     tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.MetaMessage("time_signature",
                                  numerator=beats_per_bar,
                                  denominator=beat_unit, time=0))

    tpb      = mid.ticks_per_beat
    beat_dur = 60.0 / bpm
    note_dur = int(tpb / 4)   # 16th note in ticks

    msgs: list[tuple[int, mido.Message]] = []
    for ev in events:
        tick = int(ev["time"] / beat_dur * tpb)
        vel  = min(127, max(1, int(ev["velocity"])))
        msgs.append((tick,            mido.Message("note_on",  channel=9, note=ev["note"], velocity=vel, time=0)))
        msgs.append((tick + note_dur, mido.Message("note_off", channel=9, note=ev["note"], velocity=0,   time=0)))

    msgs.sort(key=lambda x: x[0])
    prev = 0
    for abs_tick, msg in msgs:
        msg.time = abs_tick - prev
        track.append(msg)
        prev = abs_tick

    return mid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(
    file_path: str | Path,
) -> tuple[list[dict], mido.MidiFile, dict[str, Any]]:
    """
    Multi-pass drum transcription.

    Returns
    -------
    events   : list[dict]       {note, time, velocity, type, ghost, duration}
    midi     : mido.MidiFile    ready to save or upload
    metadata : dict             {bpm, time_signature, beats_per_bar,
                                 beat_unit, duration, event_count}
    """
    file_path = Path(file_path)
    log.info("Transcribing %s", file_path.name)

    # ── Load audio ───────────────────────────────────────────────────────────
    data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
    y        = data.mean(axis=1).astype(np.float32)
    duration = float(len(y)) / sr

    # ── BPM + time signature ─────────────────────────────────────────────────
    bpm, beat_times            = _detect_bpm(y, sr)
    beats_per_bar, beat_unit   = _infer_time_signature(y, sr, bpm, beat_times)
    log.info("BPM %.1f  time-sig %d/%d", bpm, beats_per_bar, beat_unit)

    # ── Initialise classifier ────────────────────────────────────────────────
    cnn, heuristic = _get_classifier()

    # ========================================================================
    # PASS 1 — Per-band onset detection
    # ========================================================================

    # Kick: 50–200 Hz
    y_kick     = _butter_filter(y, sr, "band", [50, 200])
    kick_times = _onset_times(y_kick, sr, delta=0.08, wait=8)
    kick_rms   = _rms_at(y, sr, kick_times)          # velocity from full signal
    kick_vel   = _rms_to_velocity(kick_rms, lo=50, hi=115)

    # Snare: 200–5000 Hz  (kick-bleed suppressed)
    y_snare     = _butter_filter(y, sr, "band", [200, 5000])
    snare_times = _onset_times(y_snare, sr, delta=0.06, wait=6)
    snare_times = np.array([
        t for t in snare_times
        if not any(abs(t - kt) < 0.03 for kt in kick_times)
    ])
    snare_rms = _rms_at(y, sr, snare_times)
    snare_vel = _rms_to_velocity(snare_rms, lo=35, hi=110)

    # Cymbals: 5000 Hz+
    y_cym     = _butter_filter(y, sr, "high", 5000)
    cym_times = _onset_times(y_cym, sr, delta=0.05, wait=3)
    cym_rms   = _rms_at(y, sr, cym_times)
    cym_vel   = _rms_to_velocity(cym_rms, lo=30, hi=100)

    log.info("Pass-1 onsets — kick:%d  snare:%d  cymbal:%d",
             len(kick_times), len(snare_times), len(cym_times))

    # ========================================================================
    # PASS 2 + 3 — Feature extraction & classification
    # ========================================================================

    events: list[dict] = []

    # ── Kick (no reclassification needed) ───────────────────────────────────
    for t, v in zip(kick_times, kick_vel):
        events.append({
            "note": GM_KICK, "time": round(float(t), 4),
            "velocity": int(v), "type": "kick",
            "ghost": False, "duration": 0.05,
        })

    # ── Snare / clap  (spectral sub-classification) ──────────────────────────
    for t, v in zip(snare_times, snare_vel):
        gm   = _classify_snare(float(t), y, sr, heuristic, cnn)
        name = GM_NAMES[gm]
        events.append({
            "note": gm, "time": round(float(t), 4),
            "velocity": int(v), "type": name,
            "ghost": False, "duration": 0.05,
        })

    # ── Cymbals (spectral sub-classification: HH/Ride/Crash/China) ──────────
    for t, v in zip(cym_times, cym_vel):
        gm   = _classify_cymbal(float(t), y, sr, heuristic, cnn)
        name = GM_NAMES[gm]
        events.append({
            "note": gm, "time": round(float(t), 4),
            "velocity": int(v), "type": name,
            "ghost": False, "duration": 0.05,
        })

    # ========================================================================
    # PASS 4 — Post-processing
    # ========================================================================

    events = _quantize(events, bpm, beats_per_bar)
    events = _tag_ghosts(events)
    midi   = _build_midi(events, bpm, beats_per_bar, beat_unit)

    from collections import Counter
    type_counts = Counter(e["type"] for e in events)
    log.info("Classification — %s", dict(type_counts))

    metadata: dict[str, Any] = {
        "bpm":            round(bpm, 1),
        "time_signature": f"{beats_per_bar}/{beat_unit}",
        "beats_per_bar":  beats_per_bar,
        "beat_unit":      beat_unit,
        "duration":       round(duration, 2),
        "event_count":    len(events),
    }

    log.info("Transcription complete: %d events  BPM=%.1f  %d/%d",
             len(events), bpm, beats_per_bar, beat_unit)

    return events, midi, metadata


def midi_to_bytes(midi: mido.MidiFile) -> bytes:
    buf = BytesIO()
    midi.save(file=buf)
    return buf.getvalue()
