"""
Clone Hero chart builder for drum transcriptions.

Converts transcription events (GM MIDI notes + timing) into a Clone Hero
chart zip file containing:
  - notes.chart  (the chart data)
  - song.ini     (song metadata)
  - song.ogg     (drumless mix, OGG Vorbis)
  - drums.ogg    (drums-only stem, OGG Vorbis)
"""
from __future__ import annotations

import io
import logging
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]


def _wav_to_ogg(wav_path: Path) -> bytes:
    """Convert a WAV file to OGG Vorbis using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(wav_path),
                "-c:a", "libvorbis", "-q:a", "5",
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
        )
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# GM MIDI → Clone Hero drum mapping
#
# Clone Hero ExpertDrums note numbers:
#   0  = Kick
#   1  = Red pad    (Snare)
#   2  = Yellow pad (Hi-Hat / Tom)
#   3  = Blue pad   (Tom / Cymbal)
#   4  = Green pad  (Cymbal / Tom)
#  32  = Expert+ Kick (double bass)
#  66  = Yellow cymbal marker  (pad 2 = cymbal)
#  67  = Blue cymbal marker    (pad 3 = cymbal)
#  68  = Green cymbal marker   (pad 4 = cymbal)
# ---------------------------------------------------------------------------

# Each entry: (chart_note, cymbal_marker or None)
GM_TO_CH: dict[int, tuple[int, int | None]] = {
    36: (0, None),    # Kick            → Kick
    35: (0, None),    # Acoustic Bass   → Kick
    38: (1, None),    # Snare           → Red
    40: (1, None),    # Electric Snare  → Red
    37: (1, None),    # Side Stick      → Red
    42: (2, 66),      # Closed Hi-Hat   → Yellow cymbal
    44: (2, 66),      # Pedal Hi-Hat    → Yellow cymbal
    46: (2, 66),      # Open Hi-Hat     → Yellow cymbal
    45: (3, None),    # Low Tom         → Blue tom
    43: (3, None),    # High Floor Tom  → Blue tom
    47: (3, None),    # Low-Mid Tom     → Blue tom
    48: (2, None),    # Hi-Mid Tom      → Yellow tom
    50: (2, None),    # High Tom        → Yellow tom
    49: (4, 68),      # Crash Cymbal 1  → Green cymbal
    57: (4, 68),      # Crash Cymbal 2  → Green cymbal
    51: (3, 67),      # Ride Cymbal     → Blue cymbal
    53: (3, 67),      # Ride Bell       → Blue cymbal
    52: (4, 68),      # Chinese Cymbal  → Green cymbal
    55: (4, 68),      # Splash Cymbal   → Green cymbal
}

# Fallback: map unknown notes to Red pad
_DEFAULT_CH = (1, None)

TICKS_PER_BEAT = 480


def build_chart_zip(
    events: list[dict],
    metadata: dict[str, Any],
    song_name: str = "Drum Transcription",
    song_wav: Path | None = None,
    drums_wav: Path | None = None,
    drumless_wav: Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> bytes:
    """
    Build a Clone Hero chart zip from transcription events.

    Parameters
    ----------
    events       : list of dicts with keys: note (GM MIDI), time (seconds), velocity
    metadata     : dict with bpm, beats_per_bar, beat_unit
    song_name    : display name for the chart
    song_wav     : path to the original full mix WAV (unused, kept for compat)
    drums_wav    : path to the drums-only WAV (converted to drums.ogg)
    drumless_wav : path to the drumless mix WAV (converted to song.ogg)
    on_progress  : optional callback(pct: int, message: str)

    Returns
    -------
    bytes of a zip file containing notes.chart, song.ini, song.ogg, drums.ogg
    """
    bpm = metadata.get("bpm", 120.0)
    beats_per_bar = metadata.get("beats_per_bar", 4)

    beat_dur = 60.0 / bpm
    # 2 seconds of leading silence (standard for CH charts)
    leading_ticks = int(2.0 / beat_dur * TICKS_PER_BEAT)

    # --- SyncTrack ---
    sync_lines: list[str] = []
    sync_lines.append(f"  0 = TS {beats_per_bar}")
    sync_lines.append(f"  0 = B {int(bpm * 1000)}")
    sync_track = "\n".join(sync_lines) + "\n"

    # --- ExpertDrums ---
    note_lines: list[str] = []

    for ev in sorted(events, key=lambda e: e["time"]):
        gm_note = ev["note"]
        ch_note, cymbal = GM_TO_CH.get(gm_note, _DEFAULT_CH)

        tick = leading_ticks + int(ev["time"] / beat_dur * TICKS_PER_BEAT)

        note_lines.append(f"  {tick} = N {ch_note} 0")
        if cymbal is not None:
            note_lines.append(f"  {tick} = N {cymbal} 0")

    notes_track = "\n".join(note_lines) + "\n"

    # --- notes.chart ---
    chart_content = f"""[Song]
{{
  Name = "{song_name}"
  Charter = "Oahsnail"
  Offset = 0
  Resolution = {TICKS_PER_BEAT}
  Difficulty = 4
  PreviewStart = 0
  PreviewEnd = 10000
  Genre = "rock"
  MediaType = "cd"
  MusicStream = "song.ogg"
  DrumStream = "drums.ogg"
}}
[SyncTrack]
{{
{sync_track}}}
[ExpertDrums]
{{
{notes_track}}}
"""

    # --- song.ini ---
    ini_content = f"""[Song]
name = {song_name}
artist = Unknown
album = Unknown
genre = rock
charter = drum-transcriber
diff_drums = -1
loading_phrase = Generated by drum-transcriber

"""

    # --- Build zip ---
    def _progress(pct: int, msg: str):
        if on_progress:
            on_progress(pct, msg)

    _progress(10, "Generating chart data…")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("notes.chart", chart_content)
        zf.writestr("song.ini", ini_content)

        # Convert drumless WAV → song.ogg
        if drumless_wav and drumless_wav.exists():
            _progress(20, "Converting drumless mix to OGG…")
            try:
                ogg_bytes = _wav_to_ogg(drumless_wav)
                zf.writestr("song.ogg", ogg_bytes)
                _progress(55, "song.ogg added")
            except subprocess.CalledProcessError as e:
                log.warning("Failed to convert drumless WAV to OGG: %s", e.stderr)

        # Convert drums WAV → drums.ogg
        if drums_wav and drums_wav.exists():
            _progress(60, "Converting drums stem to OGG…")
            try:
                ogg_bytes = _wav_to_ogg(drums_wav)
                zf.writestr("drums.ogg", ogg_bytes)
                _progress(95, "drums.ogg added")
            except subprocess.CalledProcessError as e:
                log.warning("Failed to convert drums WAV to OGG: %s", e.stderr)

    _progress(100, "Zip complete")
    return buf.getvalue()
