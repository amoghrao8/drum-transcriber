"""
Music Teacher service — semantic MIDI log + Qwen 2.5 Coach.

Public API:
    build_semantic_log(events, metadata) -> str
        Converts a GM MIDI event list into a human-readable measure:beat log
        describing the performance in musical terms.

    generate_lesson(job_id, semantic_log) -> str
        Sends the log to Ollama (qwen2.5:14b) with the Cat.li Coach prompt
        and returns a Markdown lesson.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from ollama import AsyncClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OLLAMA_MODEL = "qwen2.5:14b"

# Human-readable names for GM drum notes used in the log
_GM_LABELS: dict[int, str] = {
    36: "Kick",
    38: "Snare",
    39: "Clap",
    42: "HH-Closed",
    46: "HH-Open",
    49: "Crash",
    51: "Ride",
    52: "China",
}

# ---------------------------------------------------------------------------
# Velocity category helpers
# ---------------------------------------------------------------------------

def _vel_cat(v: int) -> str:
    if v >= 100: return "ff"
    if v >= 80:  return "mf"
    if v >= 55:  return "mp"
    return "p"


# ---------------------------------------------------------------------------
# Semantic log builder
# ---------------------------------------------------------------------------

def build_semantic_log(
    events: list[dict],
    metadata: dict[str, Any],
    max_measures: int = 64,
) -> str:
    """
    Convert GM MIDI events into a structured musical log.

    Example output:
        BPM: 120.0 | Time Signature: 4/4 | Duration: 185.3s | Events: 1247

        [M01 | 1   ]: Kick(ff), HH-Closed(mf)
        [M01 | 1.25]: HH-Closed(mp)
        [M01 | 1.5 ]: Kick(mf), HH-Closed(mp)
        [M01 | 2   ]: Snare(ff), HH-Closed(mf)
        ...
    """
    bpm            = float(metadata.get("bpm", 120))
    beats_per_bar  = int(metadata.get("beats_per_bar", 4))
    time_sig       = metadata.get("time_signature", "4/4")
    duration       = float(metadata.get("duration", 0))

    beat_dur       = 60.0 / bpm
    measure_dur    = beat_dur * beats_per_bar
    sixteenth_dur  = beat_dur / 4
    grid_phase     = float(metadata.get("grid_phase", 0.0))

    header = (
        f"BPM: {bpm} | Time Signature: {time_sig} | "
        f"Duration: {duration:.1f}s | Events: {len(events)}\n"
    )

    # Group events by (measure_idx, sixteenth_slot_in_measure)
    # Subtract grid_phase so measure 1 aligns with the actual beat grid origin
    grid: dict[tuple[int, int], list[dict]] = {}
    for ev in sorted(events, key=lambda e: e["time"]):
        t     = ev["time"] - grid_phase
        if t < 0:
            t = 0.0
        m_idx = int(t / measure_dur)
        if m_idx >= max_measures:
            continue
        slot_in_m = round((t % measure_dur) / sixteenth_dur)
        key = (m_idx, slot_in_m)
        grid.setdefault(key, []).append(ev)

    lines: list[str] = [header]
    prev_measure = -1

    for (m_idx, slot), slot_events in sorted(grid.items()):
        if m_idx != prev_measure:
            if prev_measure >= 0:
                lines.append("")        # blank line between measures
            prev_measure = m_idx

        # Beat position: 1-indexed beat + fractional 16th
        beat_in_bar    = slot // 4 + 1
        sub_in_beat    = slot % 4

        if sub_in_beat == 0:
            pos_str = f"{beat_in_bar}  "
        elif sub_in_beat == 1:
            pos_str = f"{beat_in_bar}.25"
        elif sub_in_beat == 2:
            pos_str = f"{beat_in_bar}.5 "
        else:
            pos_str = f"{beat_in_bar}.75"

        parts = []
        for ev in sorted(slot_events, key=lambda e: e["note"]):
            label = _GM_LABELS.get(ev["note"], f"Note{ev['note']}")
            ghost = "(ghost)" if ev.get("ghost") else ""
            parts.append(f"{label}{ghost}({_vel_cat(ev['velocity'])})")

        lines.append(f"[M{m_idx + 1:02d} | {pos_str}]: {', '.join(parts)}")

    # Add a brief per-measure density summary at the end
    lines.append("")
    lines.append("── Density summary (hits per measure) ──")
    measure_counts: dict[int, int] = {}
    for (m_idx, _), evs in grid.items():
        measure_counts[m_idx] = measure_counts.get(m_idx, 0) + len(evs)
    for m_idx in sorted(measure_counts):
        lines.append(f"  M{m_idx + 1:02d}: {measure_counts[m_idx]} hits")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Cat.li Head Coach — a world-class drum instructor with encyclopedic \
knowledge of every genre and technique.

You will receive a structured MIDI event log from a real drum performance, \
including BPM, time signature, and a measure-by-measure breakdown of every hit \
with its velocity category (ff/mf/mp/p) and ghost-note markers.

Analyse the log and respond with ONLY a Markdown document containing exactly \
these four second-level headers (copy them verbatim):

## 1. Overall Assessment
## 2. Groove Breakdown
## 3. Highlight Moments
## 4. Practice Exercises

Under each header:

## 1. Overall Assessment
State the confirmed time signature and style (Rock, Jazz, Metal, Funk, etc.). \
Describe the primary groove in 2–3 sentences. Note any unusual metric or \
stylistic features (e.g. half-time feel, polyrhythm, linear patterns).

## 2. Groove Breakdown
Analyse the recurring kick–snare relationship. Describe hi-hat subdivision, \
open-hat placement, and how ghost notes contribute to the pocket. \
Reference specific measure numbers from the log.

## 3. Highlight Moments
Identify 2–3 technically demanding or musically distinctive moments: fills, \
stack accents, complex ride patterns, metric modulations. \
Give the measure number and explain what makes each moment notable.

## 4. Practice Exercises
Generate exactly 3 exercises targeting the specific rudiments or coordination \
challenges present in this track. For each exercise:
- Give it a name
- Describe the technical goal
- Provide a step-by-step breakdown
- Suggest a starting BPM

Be specific, technical, and actionable. Reference the actual instruments and \
velocities from the log.\
"""


# ---------------------------------------------------------------------------
# Ollama lesson generator
# ---------------------------------------------------------------------------

async def generate_lesson(job_id: str, semantic_log: str) -> str:
    """
    Send the semantic log to Ollama and return the Markdown lesson.
    Raises RuntimeError if the call fails.
    """
    user_message = (
        f"Here is the drum performance log for job_id={job_id}.\n"
        f"Please analyse it and return the four-section Markdown lesson.\n\n"
        f"{semantic_log}"
    )

    try:
        client = AsyncClient()
        response = await client.chat(
            model=_OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.message.content
    except Exception as exc:
        raise RuntimeError(f"Ollama generate_lesson failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Legacy alias kept for any direct callers
# ---------------------------------------------------------------------------

def summarize_transcription(events: list[dict], duration: float = 60.0) -> str:
    """Backwards-compat shim — builds a minimal metadata dict from events."""
    # Estimate BPM from hi-hat IOIs
    hat_times = sorted(e["time"] for e in events
                       if e.get("type") in ("hat", "hihat_closed", "hihat_open")
                       and e["time"] <= duration)
    bpm = 120.0
    if len(hat_times) >= 8:
        iois = np.diff(hat_times)
        iois = iois[(iois > 0.04) & (iois < 1.5)]
        if len(iois):
            bpm = round(float(60.0 / np.median(iois) / 4 * 4), 1)
            if not (60 <= bpm <= 240):
                bpm = 120.0
    metadata = {
        "bpm": bpm,
        "time_signature": "4/4",
        "beats_per_bar": 4,
        "beat_unit": 4,
        "duration": duration,
        "event_count": len(events),
    }
    return build_semantic_log(events, metadata)
