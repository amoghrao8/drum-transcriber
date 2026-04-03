"""
MIDI-First drum transcriber — PyTorch/CUDA pipeline.

Architecture:
  1. Load drum stem WAV via soundfile
  2. Detect BPM + beat grid (librosa)
  3. Infer time signature from meter autocorrelation
  4. SPLIT-BAND onset detection — kick, snare, and hi-hat detected on
     independently filtered signals (far more accurate than single-pass FFT)
  5. Velocity estimation per hit from RMS energy
  6. Ghost-note detection: snare hits below the 30th velocity percentile
  7. Quantise all events to the nearest 16th-note grid position
  8. Export GM MIDI via mido
  9. Return structured event list + MidiFile + metadata dict

General MIDI drum channel (channel 9, 0-indexed) note map:
  36 = Kick     38 = Snare    39 = Clap
  42 = HH Closed 46 = HH Open  49 = Crash
  51 = Ride     52 = China
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
from scipy.signal import butter, sosfilt

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GM drum constants
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

# ---------------------------------------------------------------------------
# Butterworth band filters
# ---------------------------------------------------------------------------

def _butter_filter(y: np.ndarray, sr: int, ftype: str, freqs) -> np.ndarray:
    """Zero-phase Butterworth filter.  ftype: 'low' | 'high' | 'band'."""
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
    wait: int = 4,
) -> np.ndarray:
    """Return onset times in seconds using spectral flux."""
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


def _rms_at_times(y: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    """
    Compute peak RMS energy in a short window after each onset time.
    Returns raw RMS values (0-based float).
    """
    window = int(sr * 0.04)  # 40 ms
    rms = np.zeros(len(times), dtype=np.float32)
    for i, t in enumerate(times):
        s = int(t * sr)
        chunk = y[s : s + window]
        if len(chunk):
            rms[i] = float(np.sqrt(np.mean(chunk ** 2)))
    return rms


def _rms_to_velocity(rms: np.ndarray, lo: float = 30, hi: float = 110) -> np.ndarray:
    """Scale RMS values to MIDI velocity range [lo, hi]."""
    if rms.max() < 1e-8:
        return np.full(len(rms), lo, dtype=np.int32)
    norm = rms / rms.max()
    return np.clip((norm * (hi - lo) + lo).round(), lo, hi).astype(np.int32)


# ---------------------------------------------------------------------------
# BPM & beat grid
# ---------------------------------------------------------------------------

def _detect_bpm(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """Return (bpm, beat_times_seconds)."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    bpm = float(np.squeeze(tempo))
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return bpm, beat_times


# ---------------------------------------------------------------------------
# Time signature inference
# ---------------------------------------------------------------------------

def _infer_time_signature(
    y: np.ndarray,
    sr: int,
    bpm: float,
    beat_times: np.ndarray,
) -> tuple[int, int]:
    """
    Estimate beats-per-bar by looking at the strongest grouping period in the
    onset autocorrelation.  Returns (beats_per_bar, beat_unit).
    Defaults to 4/4 for ambiguous or short sequences.
    """
    if len(beat_times) < 8:
        return 4, 4

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    # Autocorrelation of onset strength
    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)

    # Convert beat duration to frames
    sr_onset = sr / 512          # onset envelope sample rate
    beat_frames = round(60.0 / bpm * sr_onset)

    # Score groupings of 2, 3, 4, 5, 6, 7
    scores: dict[int, float] = {}
    for n in (3, 4, 5, 6, 7):
        frame = beat_frames * n
        if frame < len(ac):
            scores[n] = float(ac[int(frame)])

    if not scores:
        return 4, 4

    best = max(scores, key=scores.__getitem__)
    # Only accept non-4 if its score is clearly higher
    if best != 4 and scores.get(best, 0) > scores.get(4, 0) * 1.3:
        return best, 4
    return 4, 4


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------

def _quantize(
    events: list[dict],
    bpm: float,
    beats_per_bar: int = 4,
) -> list[dict]:
    """Snap every event to the nearest 16th-note grid position."""
    grid_dur = 60.0 / bpm / 4  # 16th note in seconds
    seen: set[tuple[float, int]] = set()
    out: list[dict] = []
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
    """
    Mark snare hits whose velocity falls below the 30th percentile as ghosts.
    """
    snare_vels = [e["velocity"] for e in events if e["note"] == GM_SNARE]
    if len(snare_vels) < 4:
        return [{**e, "ghost": False} for e in events]

    threshold = float(np.percentile(snare_vels, 30))
    result = []
    for ev in events:
        if ev["note"] == GM_SNARE and ev["velocity"] < threshold:
            result.append({**ev, "ghost": True})
        else:
            result.append({**ev, "ghost": False})
    return result


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------

def _build_midi(
    events: list[dict],
    bpm: float,
    beats_per_bar: int,
    beat_unit: int,
) -> mido.MidiFile:
    """Build a type-0 MIDI file (percussion on channel 9)."""
    mid   = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo",     tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.MetaMessage("time_signature",
                                  numerator=beats_per_bar,
                                  denominator=beat_unit, time=0))

    tpb       = mid.ticks_per_beat
    beat_dur  = 60.0 / bpm
    note_dur  = int(tpb / 4)   # 16th note in ticks (for note_off)

    # Collect (abs_tick, message) pairs
    msgs: list[tuple[int, mido.Message]] = []
    for ev in events:
        abs_tick = int(ev["time"] / beat_dur * tpb)
        vel      = min(127, max(1, int(ev["velocity"])))
        msgs.append((abs_tick,
                     mido.Message("note_on",  channel=9, note=ev["note"], velocity=vel,   time=0)))
        msgs.append((abs_tick + note_dur,
                     mido.Message("note_off", channel=9, note=ev["note"], velocity=0, time=0)))

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
    Transcribe a drum-stem WAV and return:

      events   – list[dict]  {note, time, velocity, type, ghost, duration}
      midi     – mido.MidiFile  ready to save / upload
      metadata – dict  {bpm, time_signature, beats_per_bar, beat_unit, duration}
    """
    file_path = Path(file_path)
    log.info("Transcribing %s", file_path.name)

    # 1. Load audio
    data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
    y = data.mean(axis=1)                   # mono
    duration = float(len(y)) / sr

    # 2. BPM + beat grid
    bpm, beat_times = _detect_bpm(y, sr)
    log.info("BPM detected: %.1f", bpm)

    # 3. Time signature
    beats_per_bar, beat_unit = _infer_time_signature(y, sr, bpm, beat_times)
    log.info("Time signature: %d/%d", beats_per_bar, beat_unit)

    # ── 4. Split-band onset detection ──────────────────────────────────────

    # ── Kick: 50–200 Hz (fundamental thud of the bass drum) ────────────────
    y_kick = _butter_filter(y, sr, "band", [50, 200])
    kick_times = _onset_times(y_kick, sr, delta=0.08, wait=8)
    kick_rms   = _rms_at_times(y_kick, sr, kick_times)
    kick_vel   = _rms_to_velocity(kick_rms, lo=50, hi=115)

    # ── Snare: 200–5000 Hz (crack + overtones) ─────────────────────────────
    # Remove kick bleed by high-passing above 200 Hz
    y_snare = _butter_filter(y, sr, "band", [200, 5000])
    snare_times = _onset_times(y_snare, sr, delta=0.06, wait=6)
    # Suppress snare onsets within 30 ms of a kick (avoid kick bleed)
    snare_times = np.array([
        t for t in snare_times
        if not any(abs(t - kt) < 0.03 for kt in kick_times)
    ])
    snare_rms = _rms_at_times(y_snare, sr, snare_times)
    snare_vel = _rms_to_velocity(snare_rms, lo=35, hi=110)

    # ── Hi-hat: 5000 Hz+ (crisp attack of closed/open hat) ─────────────────
    y_hat = _butter_filter(y, sr, "high", 5000)
    hat_times = _onset_times(y_hat, sr, delta=0.05, wait=3)
    hat_rms   = _rms_at_times(y_hat, sr, hat_times)
    hat_vel   = _rms_to_velocity(hat_rms, lo=30, hi=100)

    # ── Open hi-hat heuristic: sustained energy after onset ────────────────
    def _is_open_hh(t: float) -> bool:
        s   = int(t * sr)
        w1  = y_hat[s : s + int(sr * 0.05)]   # first 50 ms
        w2  = y_hat[s + int(sr * 0.05) : s + int(sr * 0.15)]  # next 100 ms
        if not len(w1) or not len(w2):
            return False
        decay = np.sqrt(np.mean(w2 ** 2)) / (np.sqrt(np.mean(w1 ** 2)) + 1e-8)
        return float(decay) > 0.4   # still lots of energy → open

    # 5. Assemble raw events
    events: list[dict] = []

    for t, v in zip(kick_times, kick_vel):
        events.append({"note": GM_KICK, "time": round(float(t), 4),
                        "velocity": int(v), "type": "kick",
                        "ghost": False, "duration": 0.05})

    for t, v in zip(snare_times, snare_vel):
        events.append({"note": GM_SNARE, "time": round(float(t), 4),
                        "velocity": int(v), "type": "snare",
                        "ghost": False, "duration": 0.05})

    for t, v in zip(hat_times, hat_vel):
        gm = GM_HIHAT_OPEN if _is_open_hh(float(t)) else GM_HIHAT_CLOSED
        events.append({"note": gm, "time": round(float(t), 4),
                        "velocity": int(v), "type": GM_NAMES[gm],
                        "ghost": False, "duration": 0.05})

    log.info("Raw events — kick: %d  snare: %d  hat: %d",
             len(kick_times), len(snare_times), len(hat_times))

    # 6. Quantise to 16th-note grid
    events = _quantize(events, bpm, beats_per_bar)

    # 7. Ghost-note tagging
    events = _tag_ghosts(events)

    # 8. Build MIDI
    midi = _build_midi(events, bpm, beats_per_bar, beat_unit)

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
    """Serialise a MidiFile to raw bytes for upload."""
    buf = BytesIO()
    midi.save(file=buf)
    return buf.getvalue()
