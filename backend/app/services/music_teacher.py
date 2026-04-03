"""
Music Teacher service.

Provides two public callables:

    summarize_transcription(events, duration=60.0) -> str
        Maps the first `duration` seconds of drum events onto a human-readable
        16th-note grid.  One measure per block, four instrument rows (K S g H).

    generate_lesson(job_id, grid) -> str
        Sends the grid to a local Ollama instance (llama4:14b) and returns a
        Markdown-formatted lesson with exactly four prescribed headers.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from functools import partial
from typing import Optional

import numpy as np
from ollama import AsyncClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYMBOL: dict[str, str] = {
    "kick":        "K",
    "snare":       "S",
    "snare_ghost": "g",
    "hat":         "H",
}

_OLLAMA_MODEL = "qwen2.5:14b"

_SYSTEM_PROMPT = """\
You are a world-class drum instructor with decades of experience teaching \
professional drummers across every genre. A student has given you an \
automatically-generated 16th-note grid of a drum performance. Each character \
represents one sixteenth note: K=Kick, S=Snare, g=Ghost note, H=Hi-hat, \
.=Rest. Beat boundaries are marked with |.

Analyse the grid carefully and respond with ONLY a Markdown document that \
contains exactly these four second-level headers (copy them verbatim):

## 1. Overall Transcription:
## 2. Key Rudiments:
## 3. Ghost Note Mastery:
## 4. Groove & Coordination:

Under each header write the following:

## 1. Overall Transcription:
Identify the time signature (look closely for 7/8 phrasing), calculate the \
exact tempo from the grid metadata, and characterise the feel as either \
'linear' (instruments rarely stack) or 'pocket' (kick/snare anchor with dense \
hat activity).

## 2. Key Rudiments:
Identify any sticking rudiments implied by the snare and ghost-note pattern \
(e.g. Paradiddle, Double Stroke Roll, Flam Tap). Provide exactly 2 written \
practice exercises with a short description of each.

## 3. Ghost Note Mastery:
Analyse the ghost-note density and dynamics relative to the full-velocity \
snare hits. Provide exactly 2 practical exercises for developing 'chatter' \
(ghost-note subtlety and control).

## 4. Groove & Coordination:
Break down the kick-snare rhythmic relationship specifically in the context of \
the time signature identified. Provide 1 limb-independence exercise that \
isolates the coordination challenge present in this transcription.

Be specific, technical, and actionable. Reference bar numbers from the grid \
where relevant.\
"""


# ---------------------------------------------------------------------------
# BPM estimation
# ---------------------------------------------------------------------------

def _estimate_bpm(events: list[dict], max_time: float) -> tuple[float, float]:
    """
    Returns (bpm, sixteenth_note_duration_seconds).

    Strategy: use hi-hat onsets (most pulse-regular) when >= 8 are present,
    otherwise fall back to all onsets.  Modal IOI in 5 ms bins → scale to the
    nearest musical subdivision that puts BPM in [60, 240].
    """
    hat_times = sorted(
        e["time"] for e in events
        if e["time"] <= max_time and e["type"] == "hat"
    )
    all_times = sorted(e["time"] for e in events if e["time"] <= max_time)
    times = hat_times if len(hat_times) >= 8 else all_times

    if len(times) < 4:
        return 120.0, 60.0 / 480.0

    iois = np.diff(times)
    iois = iois[(iois >= 0.04) & (iois <= 1.5)]
    if not len(iois):
        return 120.0, 60.0 / 480.0

    # Modal IOI (5 ms resolution)
    bins = np.round(iois / 0.005).astype(int)
    modal_bin = Counter(bins.tolist()).most_common(1)[0][0]
    modal_ioi = modal_bin * 0.005

    # Try interpreting the modal IOI as 16th, 8th, or quarter note
    for divisor in (4, 2, 1):
        bpm = 60.0 / (divisor * modal_ioi)
        if 60.0 <= bpm <= 240.0:
            sixteenth = 60.0 / (bpm * 4)
            return round(bpm, 1), sixteenth

    return 120.0, 60.0 / 480.0


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def summarize_transcription(
    events: list[dict],
    duration: float = 60.0,
) -> str:
    """
    Convert raw drum events (list of {time, type, velocity}) into a
    human-readable 16th-note grid covering the first `duration` seconds.

    Layout per measure:
        M01:  K |X...|....|X...|....|
              S |....|X...|....|X...|
              g |..X.|..X.|..X.|..X.|
              H |X.X.|X.X.|X.X.|X.X.|
    """
    events_window = [e for e in events if e["time"] <= duration]
    if not events_window:
        return "(no events in the first 60 seconds)"

    bpm, sixteenth = _estimate_bpm(events_window, duration)

    total_slots = max(int(duration / sixteenth) + 1, 1)

    # Map each event to the nearest 16th-note slot
    slot_hits: dict[int, list[str]] = {i: [] for i in range(total_slots)}
    for ev in events_window:
        slot = round(ev["time"] / sixteenth)
        if slot < total_slots:
            sym = _SYMBOL.get(ev["type"], "?")
            slot_hits[slot].append(sym)

    # Build rows: 16 slots per measure (4/4 display grouping)
    slots_per_measure = 16
    slots_per_beat = 4
    total_measures = (total_slots + slots_per_measure - 1) // slots_per_measure

    lines: list[str] = [
        f"BPM ~{bpm}  |  K=Kick  S=Snare  g=Ghost  H=Hi-hat",
        "1 character = 1 sixteenth note  |  = beat boundary",
        "",
    ]

    for m in range(total_measures):
        base = m * slots_per_measure
        measure_num = f"M{m + 1:02d}"

        instrument_rows: list[str] = []
        for sym, label in [("K", "K"), ("S", "S"), ("g", "g"), ("H", "H")]:
            row = ""
            for beat in range(slots_per_measure // slots_per_beat):
                row += "|"
                for pos in range(slots_per_beat):
                    slot = base + beat * slots_per_beat + pos
                    hits = slot_hits.get(slot, [])
                    row += sym if sym in hits else "."
            row += "|"
            indent = "       " if label != "K" else f"{measure_num}:  "
            instrument_rows.append(f"{indent}{label} {row}")

        lines.extend(instrument_rows)
        lines.append("")  # blank line between measures

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama lesson generator
# ---------------------------------------------------------------------------

async def generate_lesson(job_id: str, grid: str) -> str:
    """
    Send the 16th-note grid to a local Ollama instance and return the
    Markdown lesson string.

    Raises RuntimeError if the Ollama call fails.
    """
    user_message = (
        f"Here is the drum transcription grid for job_id={job_id}.\n"
        f"Please analyse it and return the four-section Markdown lesson.\n\n"
        f"{grid}"
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
