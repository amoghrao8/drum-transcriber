"""
ADTOF drum transcriber.

Pipeline
────────
Step 1  Load drum stem WAV
Step 2  Detect BPM + time signature (librosa)
Step 3  ADTOF inference → {kick, snare, toms, hihat, cymbal} onset times
Step 4  Velocity from drum stem RMS at each onset
Step 5  Ghost-note tagging (snare < 20th-pct velocity)
Step 6  16th-note quantisation anchored to detected beat phase
Step 7  GM MIDI export (mido)

GM MIDI percussion channel 9:
  36 Kick   38 Snare   42 Hi-Hat   45 Tom   49 Cymbal
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

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GM drum constants
# ---------------------------------------------------------------------------
GM_KICK   = 36
GM_SNARE  = 38
GM_HIHAT  = 42
GM_TOM    = 45
GM_CYMBAL = 49

GM_NAMES: dict[int, str] = {
    GM_KICK:   "kick",
    GM_SNARE:  "snare",
    GM_HIHAT:  "hihat",
    GM_TOM:    "tom",
    GM_CYMBAL: "cymbal",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_at(y: np.ndarray, sr: int, times: np.ndarray, window_s: float = 0.04) -> np.ndarray:
    w   = int(sr * window_s)
    rms = np.zeros(len(times), dtype=np.float32)
    for i, t in enumerate(times):
        s     = int(t * sr)
        chunk = y[s : s + w]
        if len(chunk):
            rms[i] = float(np.sqrt(np.mean(chunk ** 2)))
    return rms


def _rms_to_velocity(rms: np.ndarray, lo: int = 30, hi: int = 115) -> np.ndarray:
    if len(rms) == 0:
        return np.array([], dtype=np.int32)
    if rms.max() < 1e-8:
        return np.full(len(rms), lo, dtype=np.int32)
    norm = rms / rms.max()
    return np.clip((norm * (hi - lo) + lo).round(), lo, hi).astype(np.int32)


# ---------------------------------------------------------------------------
# BPM + time signature
# ---------------------------------------------------------------------------

def _detect_bpm(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames", start_bpm=100)
    bpm = float(np.squeeze(tempo))

    while bpm > 200:
        bpm /= 2
    while bpm < 60:
        bpm *= 2

    # Librosa sometimes locks onto 4/3× the true tempo; correct it
    candidate_34 = bpm * 0.75
    if 115 <= bpm <= 160 and 80 <= candidate_34 <= 115:
        bpm = candidate_34

    return bpm, librosa.frames_to_time(beat_frames, sr=sr)


def _infer_time_signature(
    y: np.ndarray, sr: int, bpm: float, beat_times: np.ndarray
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
# Post-processing
# ---------------------------------------------------------------------------

def _tag_ghosts(events: list[dict]) -> list[dict]:
    snare_vels = [e["velocity"] for e in events if e["note"] == GM_SNARE]
    if len(snare_vels) < 4:
        return [{**e, "ghost": False} for e in events]
    threshold = float(np.percentile(snare_vels, 20))
    return [
        {**ev, "ghost": ev["note"] == GM_SNARE and ev["velocity"] < threshold}
        for ev in events
    ]


def _quantize(
    events:        list[dict],
    bpm:           float,
    beats_per_bar: int = 4,
    subdivisions:  int = 4,
    beat_times:    np.ndarray | None = None,
) -> tuple[list[dict], float, float]:
    """
    Quantize to a uniform phase-anchored 16th-note grid.
    Returns (events, grid_dur, phase).
    """
    if beat_times is not None and len(beat_times) >= 2:
        ibi      = float(np.median(np.diff(beat_times)))
        grid_dur = ibi / subdivisions
        phase    = beat_times[0] % grid_dur
    else:
        grid_dur = 60.0 / bpm / subdivisions
        phase    = 0.0

    seen: set[tuple[float, int]] = set()
    out:  list[dict]             = []
    for ev in sorted(events, key=lambda e: e["time"]):
        idx    = round((ev["time"] - phase) / grid_dur)
        q_time = round(phase + idx * grid_dur, 5)
        key    = (q_time, ev["note"])
        if key not in seen:
            seen.add(key)
            out.append({**ev, "time": q_time})

    return out, grid_dur, phase


def _build_midi(
    events: list[dict], bpm: float, beats_per_bar: int, beat_unit: int
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
    note_dur = int(tpb / 4)   # 16th note

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
    *,
    override_bpm: float | None = None,
    override_beats_per_bar: int | None = None,
    override_beat_unit: int | None = None,
    quantize: bool = False,
) -> tuple[list[dict], mido.MidiFile, dict[str, Any]]:
    """
    Transcribe a drum-stem WAV using ADTOF neural drum transcription.

    Parameters
    ----------
    override_bpm           : If set, skip auto BPM detection and use this value.
    override_beats_per_bar : If set, override the numerator of the time signature.
    override_beat_unit     : If set, override the denominator of the time signature.
    quantize               : If False, skip 16th-note grid quantisation.

    Returns
    -------
    events   : list[dict]     {note, time, velocity, type, ghost, duration}
    midi     : mido.MidiFile
    metadata : dict
    """
    file_path = Path(file_path)
    log.info("Transcribing %s", file_path.name)

    # ── Step 1: Load audio ───────────────────────────────────────────────────
    data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
    y        = data.mean(axis=1).astype(np.float32)
    duration = float(len(y)) / sr

    # ── Step 2: BPM + time signature ─────────────────────────────────────────
    bpm, beat_times          = _detect_bpm(y, sr)
    beats_per_bar, beat_unit = _infer_time_signature(y, sr, bpm, beat_times)

    # Apply manual overrides (if provided)
    if override_bpm is not None:
        bpm = override_bpm
        # Recompute beat_times from the override BPM so quantisation grid is correct
        beat_dur = 60.0 / bpm
        beat_times = np.arange(beat_times[0] if len(beat_times) else 0.0, duration, beat_dur)
    if override_beats_per_bar is not None:
        beats_per_bar = override_beats_per_bar
    if override_beat_unit is not None:
        beat_unit = override_beat_unit

    log.info("BPM %.1f  %d/%d%s", bpm, beats_per_bar, beat_unit,
             "  (manual override)" if override_bpm or override_beats_per_bar or override_beat_unit else "")

    # ── Step 3: ADTOF inference ───────────────────────────────────────────────
    from app.models.adtof.transcriber import transcribe_stem
    log.info("Running ADTOF…")
    hits = transcribe_stem(str(file_path))
    # hits = {36: [times], 38: [times], 42: [times], 45: [times], 49: [times]}

    log.info("ADTOF hits — kick:%d snare:%d hihat:%d toms:%d cymbal:%d",
             len(hits.get(GM_KICK, [])),  len(hits.get(GM_SNARE, [])),
             len(hits.get(GM_HIHAT, [])), len(hits.get(GM_TOM, [])),
             len(hits.get(GM_CYMBAL, [])))

    # ── Step 4: Velocity from drum stem RMS ──────────────────────────────────
    vel_ranges = {
        GM_KICK:   (50, 115),
        GM_SNARE:  (35, 110),
        GM_HIHAT:  (25, 95),
        GM_TOM:    (40, 110),
        GM_CYMBAL: (40, 110),
    }

    events: list[dict] = []
    type_names = {GM_KICK: "kick", GM_SNARE: "snare", GM_HIHAT: "hihat",
                  GM_TOM: "tom", GM_CYMBAL: "cymbal"}

    for gm_note, times_list in hits.items():
        if not times_list:
            continue
        times = np.array(times_list, dtype=np.float32)
        lo, hi = vel_ranges.get(gm_note, (30, 110))
        vels   = _rms_to_velocity(_rms_at(y, sr, times), lo=lo, hi=hi)
        for t, v in zip(times, vels):
            events.append({
                "note":     gm_note,
                "time":     round(float(t), 4),
                "velocity": int(v),
                "type":     type_names.get(gm_note, "unknown"),
                "ghost":    False,
                "duration": 0.05,
            })

    # ── Steps 5-7: Post-processing ────────────────────────────────────────────
    grid_dur = 60.0 / bpm / 4
    phase    = 0.0
    if quantize:
        events, grid_dur, phase = _quantize(events, bpm, beats_per_bar,
                                            subdivisions=4, beat_times=beat_times)
    events = _tag_ghosts(events)
    midi   = _build_midi(events, bpm, beats_per_bar, beat_unit)

    from collections import Counter
    log.info("Result: %s", dict(Counter(e["type"] for e in events)))

    metadata: dict[str, Any] = {
        "bpm":               round(bpm, 1),
        "time_signature":    f"{beats_per_bar}/{beat_unit}",
        "beats_per_bar":     beats_per_bar,
        "beat_unit":         beat_unit,
        "duration":          round(duration, 2),
        "event_count":       len(events),
        "quantized":         quantize,
        "grid_phase":        round(float(phase), 6),
        "grid_subdivisions": 4,
    }

    log.info("Transcription complete: %d events  BPM=%.1f", len(events), bpm)
    return events, midi, metadata


def midi_to_bytes(midi: mido.MidiFile) -> bytes:
    buf = BytesIO()
    midi.save(file=buf)
    return buf.getvalue()
