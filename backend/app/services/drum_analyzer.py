"""
Dual MDX23C-DrumSep drum transcriber.

Pipeline
────────
Step 1  Load drum stem WAV
Step 2  Detect BPM + time signature (librosa)
Step 3a Run jarredou 5-stem → kick | snare | toms | hh | cymbals
        use:  kick, snare, toms, hh  (higher SDR across all four)
Step 3b Run aufr33/jarredou 6-stem → kick | snare | toms | hh | ride | crash
        use:  ride + crash  (summed → single cymbal stem, separate from hh)
Step 4  Save all stems to stems/separated/{stem_id}/
Step 5  Per-stem onset detection on 5 voices
Step 6  Cross-validate: snare↔hh, toms↔kick
Step 7  Velocity from original mix RMS
Step 8  GM mapping: kick→36  snare→38  toms→45  hh→42  cymbal→49
Step 9  Ghost-note tagging (snare < 20th-pct velocity)
Step 10 32nd-note quantisation + deduplication
Step 11 GM MIDI export (mido)

GM MIDI percussion channel 9:
  36 Kick   38 Snare   42 Hi-Hat   45 Tom   49 Cymbal (ride+crash)
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

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GM drum constants
# ---------------------------------------------------------------------------
GM_KICK  = 36
GM_SNARE = 38
GM_HIHAT = 42
GM_TOM   = 45
GM_CRASH = 49
GM_RIDE  = 51

GM_NAMES: dict[int, str] = {
    GM_KICK:  "kick",
    GM_SNARE: "snare",
    GM_HIHAT: "hihat",
    GM_TOM:   "tom",
    GM_CRASH: "crash",
    GM_RIDE:  "ride",
}

# ---------------------------------------------------------------------------
# Lazy model singletons
# ---------------------------------------------------------------------------
_mdx5_model  = _mdx5_config  = _mdx5_device  = None
_mdx6_model  = _mdx6_config  = _mdx6_device  = None


def _get_5stem():
    global _mdx5_model, _mdx5_config, _mdx5_device
    if _mdx5_model is None:
        from app.models.drumsep.separator import load_5stem_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading jarredou 5-stem MDX23C on device=%s", device)
        _mdx5_model, _mdx5_config, _mdx5_device = load_5stem_model(device=device)
    return _mdx5_model, _mdx5_config, _mdx5_device


def _get_6stem():
    global _mdx6_model, _mdx6_config, _mdx6_device
    if _mdx6_model is None:
        from app.models.drumsep.separator import load_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading aufr33/jarredou 6-stem MDX23C on device=%s", device)
        _mdx6_model, _mdx6_config, _mdx6_device = load_model(device=device)
    return _mdx6_model, _mdx6_config, _mdx6_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onset_times(
    y: np.ndarray,
    sr: int,
    *,
    delta: float,
    wait: int,
    hop_length: int = 256,
    backtrack: bool = True,
    min_rms: float = 0.0,
) -> np.ndarray:
    frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        backtrack=backtrack,
        units="frames",
        pre_max=3, post_max=3,
        pre_avg=3, post_avg=5,
        delta=delta,
        wait=wait,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    # Absolute amplitude gate — reject onsets where the stem is essentially silent.
    # Librosa onset detection is relative within the stem; without this gate,
    # it fires on bleed/noise even when no real hit occurred.
    if min_rms > 0.0 and len(times) > 0:
        rms = _rms_at(y, sr, times, window_s=0.04)
        times = times[rms >= min_rms]

    return times


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


def _cross_validate(
    primary_times:    np.ndarray,
    primary_audio:    np.ndarray,
    competitor_times: np.ndarray,
    competitor_audio: np.ndarray,
    sr:               int,
    tol_s:            float = 0.015,
) -> np.ndarray:
    """Remove primary onsets dominated by a competitor stem at the same time."""
    if len(primary_times) == 0 or len(competitor_times) == 0:
        return primary_times
    keep = np.ones(len(primary_times), dtype=bool)
    for i, t in enumerate(primary_times):
        nearby = competitor_times[np.abs(competitor_times - t) < tol_s]
        if len(nearby):
            p_rms = _rms_at(primary_audio,    sr, np.array([t]),         0.04)[0]
            c_rms = _rms_at(competitor_audio, sr, np.array([nearby[0]]), 0.04)[0]
            if c_rms > p_rms:
                keep[i] = False
    return primary_times[keep]


# ---------------------------------------------------------------------------
# Cymbal classifier — crash vs ride by onset amplitude
# ---------------------------------------------------------------------------

def _classify_cymbals(
    times:       np.ndarray,
    y_cymbal:    np.ndarray,
    sr:          int,
    crash_ratio: float = 2.0,
    window_s:    float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split cymbal onset times into crash and ride by amplitude.

    Crashes have a much larger transient than ride bow-hits, typically 2–4×
    the median onset RMS across all cymbal events.

    Returns (crash_times, ride_times).
    """
    if len(times) == 0:
        return np.array([]), np.array([])

    rms = _rms_at(y_cymbal, sr, times, window_s=window_s)
    median_rms = float(np.median(rms))

    if median_rms < 1e-8:
        return np.array([]), times.copy()

    is_crash = rms > crash_ratio * median_rms
    return times[is_crash], times[~is_crash]


# ---------------------------------------------------------------------------
# BPM + time signature
# ---------------------------------------------------------------------------

def _detect_bpm(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    # Use start_bpm=100 to bias toward typical rock/pop tempos.
    # librosa commonly returns 4/3× the true tempo (e.g. 129 instead of 97);
    # the sanity check below corrects that.
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames", start_bpm=100)
    bpm = float(np.squeeze(tempo))

    # Clamp to musical range
    while bpm > 200:
        bpm /= 2
    while bpm < 60:
        bpm *= 2

    # If BPM is in the 115–160 range, check whether 3/4 × BPM lands in
    # a more common tempo region (80–115). Librosa sometimes locks onto
    # the triplet-subdivision pulse instead of the quarter-note beat.
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
) -> list[dict]:
    """
    Quantize events to the nearest subdivision grid slot.

    When beat_times are provided (from librosa.beat.beat_track), each event
    is anchored to its nearest detected beat and then subdivided within that
    beat. This corrects for tempo drift and pickup-bar offsets — both of which
    cause a fixed-BPM grid from t=0 to drift out of alignment.

    Falls back to a fixed BPM grid if beat_times is None or too short.
    """
    seen: set[tuple[float, int]] = set()
    out:  list[dict]             = []

    if beat_times is not None and len(beat_times) >= 2:
        # Median inter-beat interval — more stable than BPM arithmetic
        ibi = float(np.median(np.diff(beat_times)))
        subdiv_dur = ibi / subdivisions

        for ev in sorted(events, key=lambda e: e["time"]):
            t = ev["time"]
            # Nearest detected beat
            idx_beat  = int(np.argmin(np.abs(beat_times - t)))
            beat_t    = beat_times[idx_beat]
            # Subdivide around that beat
            offset    = t - beat_t
            sub_idx   = round(offset / subdiv_dur)
            q_time    = round(beat_t + sub_idx * subdiv_dur, 5)
            key       = (q_time, ev["note"])
            if key not in seen:
                seen.add(key)
                out.append({**ev, "time": q_time})
    else:
        grid_dur = 60.0 / bpm / subdivisions
        for ev in sorted(events, key=lambda e: e["time"]):
            idx    = round(ev["time"] / grid_dur)
            q_time = round(idx * grid_dur, 5)
            key    = (q_time, ev["note"])
            if key not in seen:
                seen.add(key)
                out.append({**ev, "time": q_time})

    return out


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
    note_dur = int(tpb / 8)   # 32nd note

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
    Transcribe a drum-stem WAV using dual MDX23C-DrumSep separation.

    Returns
    -------
    events   : list[dict]     {note, time, velocity, type, ghost, duration}
    midi     : mido.MidiFile
    metadata : dict           {bpm, time_signature, beats_per_bar, beat_unit,
                               duration, event_count}
    """
    file_path = Path(file_path)
    log.info("Transcribing %s", file_path.name)

    # ── Step 1: Load original audio ─────────────────────────────────────────
    data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
    y        = data.mean(axis=1).astype(np.float32)
    duration = float(len(y)) / sr

    # ── Step 2: BPM + time signature ────────────────────────────────────────
    bpm, beat_times          = _detect_bpm(y, sr)
    beats_per_bar, beat_unit = _infer_time_signature(y, sr, bpm, beat_times)
    log.info("BPM %.1f  %d/%d", bpm, beats_per_bar, beat_unit)

    from app.models.drumsep.separator import separate

    # ── Step 3a: 5-stem (kick/snare/toms/hh — higher SDR) ───────────────────
    m5, c5, d5 = _get_5stem()
    log.info("Running 5-stem separation…")
    stems5 = separate(str(file_path), m5, c5, d5)
    y_kick  = stems5["kick"]
    y_snare = stems5["snare"]
    y_toms  = stems5["toms"]
    y_hh    = stems5["hh"]

    # ── Step 3b: 6-stem (ride + crash → cymbal stem) ─────────────────────────
    m6, c6, d6 = _get_6stem()
    log.info("Running 6-stem separation…")
    stems6   = separate(str(file_path), m6, c6, d6)
    y_cymbal = stems6["ride"] + stems6["crash"]

    # ── Step 4: Save separated stems ────────────────────────────────────────
    stem_id = file_path.stem
    sep_dir = file_path.parent / "separated" / stem_id
    sep_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in stems5.items():
        sf.write(str(sep_dir / f"5stem_{name}.wav"), arr, sr)
    for name in ("ride", "crash", "hh"):
        sf.write(str(sep_dir / f"6stem_{name}.wav"), stems6[name], sr)
    sf.write(str(sep_dir / "cymbal_merged.wav"), y_cymbal, sr)
    log.info("Separated stems saved to %s", sep_dir)

    log.info("Detecting onsets…")

    # ── Step 5: Per-stem onset detection ────────────────────────────────────
    # Absolute RMS floors derived from the mix, not the stem.
    # A stem onset only counts if its RMS at that moment is at least
    # `stem_ratio` × the mix RMS, ensuring the stem dominates the mix energy.
    # Toms and cymbals get a stricter ratio because they suffer more bleed.
    mix_rms_global = float(np.sqrt(np.mean(y ** 2)))

    kick_times   = _onset_times(y_kick,   sr, delta=0.04, wait=10, hop_length=256,
                                min_rms=mix_rms_global * 0.05)
    snare_times  = _onset_times(y_snare,  sr, delta=0.06, wait=12, hop_length=256,
                                min_rms=mix_rms_global * 0.05)
    tom_times    = _onset_times(y_toms,   sr, delta=0.12, wait=20, hop_length=256,
                                min_rms=mix_rms_global * 0.20)
    hh_times     = _onset_times(y_hh,     sr, delta=0.02, wait=4,  hop_length=128,
                                backtrack=False, min_rms=mix_rms_global * 0.01)
    cymbal_times = _onset_times(y_cymbal, sr, delta=0.10, wait=10, hop_length=128,
                                backtrack=False, min_rms=mix_rms_global * 0.20)

    log.info("Onsets — kick:%d snare:%d toms:%d hh:%d cymbal:%d",
             len(kick_times), len(snare_times), len(tom_times),
             len(hh_times), len(cymbal_times))

    # ── Step 6: Cross-validation ─────────────────────────────────────────────
    # snare vs kick: kick bleed into snare stem fires false snare at kick times
    snare_times = _cross_validate(snare_times, y_snare, kick_times,  y_kick,  sr)
    snare_times = _cross_validate(snare_times, y_snare, hh_times,    y_hh,    sr)
    hh_times    = _cross_validate(hh_times,    y_hh,    snare_times, y_snare, sr)
    tom_times   = _cross_validate(tom_times,   y_toms,  kick_times,  y_kick,  sr)
    tom_times   = _cross_validate(tom_times,   y_toms,  hh_times,    y_hh,    sr)

    # ── Step 6b: Classify cymbals → crash vs ride ────────────────────────────
    crash_times, ride_times = _classify_cymbals(cymbal_times, y_cymbal, sr, crash_ratio=2.0)

    log.info("After cross-val — snare:%d hh:%d toms:%d  cymbal→crash:%d ride:%d",
             len(snare_times), len(hh_times), len(tom_times),
             len(crash_times), len(ride_times))

    # ── Step 7: Velocity from original mix RMS ───────────────────────────────
    kick_vel  = _rms_to_velocity(_rms_at(y, sr, kick_times),   lo=50, hi=115)
    snare_vel = _rms_to_velocity(_rms_at(y, sr, snare_times),  lo=35, hi=110)
    tom_vel   = _rms_to_velocity(_rms_at(y, sr, tom_times),    lo=40, hi=110)
    hh_vel    = _rms_to_velocity(_rms_at(y, sr, hh_times),     lo=25, hi=95)
    crash_vel = _rms_to_velocity(_rms_at(y, sr, crash_times),  lo=50, hi=115)
    ride_vel  = _rms_to_velocity(_rms_at(y, sr, ride_times),   lo=30, hi=100)

    # ── Step 8: Assemble events ──────────────────────────────────────────────
    events: list[dict] = []

    for t, v in zip(kick_times, kick_vel):
        events.append({"note": GM_KICK,  "time": round(float(t), 4),
                       "velocity": int(v), "type": "kick",
                       "ghost": False, "duration": 0.05})
    for t, v in zip(snare_times, snare_vel):
        events.append({"note": GM_SNARE, "time": round(float(t), 4),
                       "velocity": int(v), "type": "snare",
                       "ghost": False, "duration": 0.05})
    for t, v in zip(tom_times, tom_vel):
        events.append({"note": GM_TOM,   "time": round(float(t), 4),
                       "velocity": int(v), "type": "tom",
                       "ghost": False, "duration": 0.05})
    for t, v in zip(hh_times, hh_vel):
        events.append({"note": GM_HIHAT, "time": round(float(t), 4),
                       "velocity": int(v), "type": "hihat",
                       "ghost": False, "duration": 0.05})
    for t, v in zip(crash_times, crash_vel):
        events.append({"note": GM_CRASH, "time": round(float(t), 4),
                       "velocity": int(v), "type": "crash",
                       "ghost": False, "duration": 0.05})
    for t, v in zip(ride_times, ride_vel):
        events.append({"note": GM_RIDE,  "time": round(float(t), 4),
                       "velocity": int(v), "type": "ride",
                       "ghost": False, "duration": 0.05})

    # ── Steps 9-11: Post-processing ──────────────────────────────────────────
    events = _quantize(events, bpm, beats_per_bar, subdivisions=4, beat_times=beat_times)
    events = _tag_ghosts(events)
    midi   = _build_midi(events, bpm, beats_per_bar, beat_unit)

    from collections import Counter
    log.info("Result: %s", dict(Counter(e["type"] for e in events)))

    metadata: dict[str, Any] = {
        "bpm":            round(bpm, 1),
        "time_signature": f"{beats_per_bar}/{beat_unit}",
        "beats_per_bar":  beats_per_bar,
        "beat_unit":      beat_unit,
        "duration":       round(duration, 2),
        "event_count":    len(events),
    }

    log.info("Transcription complete: %d events  BPM=%.1f", len(events), bpm)
    return events, midi, metadata


def midi_to_bytes(midi: mido.MidiFile) -> bytes:
    buf = BytesIO()
    midi.save(file=buf)
    return buf.getvalue()
