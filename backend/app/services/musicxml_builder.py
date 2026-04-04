"""
Build a MusicXML 3.1 score from stored drum events + metadata.

No extra dependencies — generates XML as a string directly.

Two-voice layout (standard drum notation):
  Voice 1  stems-up   — SD, HH, TOM, CRASH/RIDE
  Voice 2  stems-down — BD

Ghost notes are rendered with parenthesised noteheads.
"""
from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Percussion staff positions + notehead style per GM note
# Positions follow the treble-clef visual grid that OSMD uses for percussion:
#   lines (bottom→top): E4 G4 B4 D5 F5
#   spaces: F4 A4 C5 E5
#   above top line: G5 A5 B5 C6 …
#   below bottom line: D4 C4 B3 …
# ---------------------------------------------------------------------------
_PERC: dict[int, tuple[str, int, str, str]] = {
    # note: (display-step, display-octave, stem, notehead)
    36: ('C', 4, 'down', 'normal'),   # BD  — ledger line below staff
    38: ('C', 5, 'up',   'normal'),   # SD  — 3rd space (middle of staff)
    42: ('G', 5, 'up',   'x'),        # HH  — 1st space above top line
    45: ('A', 4, 'down', 'normal'),   # TOM — 2nd space
    49: ('A', 5, 'up',   'x'),        # Crash/Ride — above HH
}

_VOICE1 = {38, 42, 45, 49}   # stems up
_VOICE2 = {36}               # stems down (BD)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _note_el(
    step: str,
    octave: int,
    stem: str,
    head: str,
    voice: int,
    is_chord: bool = False,
    ghost: bool = False,
) -> ET.Element:
    note = ET.Element('note')
    if is_chord:
        ET.SubElement(note, 'chord')
    unp = ET.SubElement(note, 'unpitched')
    ET.SubElement(unp, 'display-step').text   = step
    ET.SubElement(unp, 'display-octave').text = str(octave)
    ET.SubElement(note, 'duration').text      = '1'   # 1 division = 16th note
    ET.SubElement(note, 'voice').text         = str(voice)
    dur = ET.SubElement(note, 'type')
    dur.text = '16th'
    ET.SubElement(note, 'stem').text          = stem
    nh = ET.SubElement(note, 'notehead')
    nh.text = head
    if ghost:
        nh.set('parentheses', 'yes')
    return note


def _rest_el(voice: int) -> ET.Element:
    note = ET.Element('note')
    ET.SubElement(note, 'rest')
    ET.SubElement(note, 'duration').text = '1'
    ET.SubElement(note, 'voice').text    = str(voice)
    ET.SubElement(note, 'type').text     = '16th'
    return note


def _backup_el(divisions: int) -> ET.Element:
    b = ET.Element('backup')
    ET.SubElement(b, 'duration').text = str(divisions)
    return b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_musicxml(
    events:   list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    """
    Convert quantized drum events + metadata into a MusicXML 3.1 string.

    Parameters
    ----------
    events   : list of {note, time, velocity, ghost, ...}
    metadata : {bpm, beats_per_bar, beat_unit, grid_phase, ...}

    Returns
    -------
    MusicXML document as a UTF-8 string.
    """
    bpm           = float(metadata.get('bpm',           120))
    beats_per_bar = int(  metadata.get('beats_per_bar',   4))
    beat_unit     = int(  metadata.get('beat_unit',        4))
    grid_phase    = float(metadata.get('grid_phase',     0.0))

    grid_unit        = 60.0 / bpm / 4          # 16th-note duration in seconds
    slots_per_measure = beats_per_bar * 4       # 16 slots for 4/4

    # ── Group events by (measure, slot) ─────────────────────────────────────
    # slot_data[m_idx][slot] = list of (gm_note, ghost)
    slot_data: dict[int, dict[int, list[tuple[int, bool]]]] = {}

    for ev in events:
        gm   = int(ev['note'])
        if gm not in _PERC:
            continue
        t        = float(ev['time'])
        ghost    = bool(ev.get('ghost', False))
        abs_slot = round((t - grid_phase) / grid_unit)
        if abs_slot < 0:
            abs_slot = 0
        m_idx  = abs_slot // slots_per_measure
        s_idx  = abs_slot %  slots_per_measure
        slot_data.setdefault(m_idx, {}).setdefault(s_idx, []).append((gm, ghost))

    total_measures = (max(slot_data.keys()) + 2) if slot_data else 1

    # ── Build XML tree ───────────────────────────────────────────────────────
    root = ET.Element('score-partwise', version='3.1')

    # Part list
    part_list = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id='P1')
    ET.SubElement(score_part, 'part-name').text = 'Drumset'

    part = ET.SubElement(root, 'part', id='P1')

    for m_idx in range(total_measures):
        measure = ET.SubElement(part, 'measure', number=str(m_idx + 1))
        slots   = slot_data.get(m_idx, {})

        # Attributes (only in measure 1)
        if m_idx == 0:
            attrs = ET.SubElement(measure, 'attributes')
            ET.SubElement(attrs, 'divisions').text = '4'   # per quarter note
            key = ET.SubElement(attrs, 'key')
            ET.SubElement(key, 'fifths').text = '0'
            time_el = ET.SubElement(attrs, 'time')
            ET.SubElement(time_el, 'beats').text     = str(beats_per_bar)
            ET.SubElement(time_el, 'beat-type').text = str(beat_unit)
            clef_el = ET.SubElement(attrs, 'clef')
            ET.SubElement(clef_el, 'sign').text = 'percussion'

            # Tempo direction
            direction = ET.SubElement(measure, 'direction', placement='above')
            dt = ET.SubElement(direction, 'direction-type')
            metro = ET.SubElement(dt, 'metronome', parentheses='no')
            ET.SubElement(metro, 'beat-unit').text   = 'quarter'
            ET.SubElement(metro, 'per-minute').text  = str(int(round(bpm)))
            ET.SubElement(direction, 'sound', tempo=str(int(round(bpm))))

        # ── Voice 1 (stems-up: SD HH TOM CYM) ──────────────────────────────
        for s in range(slots_per_measure):
            hits = [(gm, gh) for gm, gh in slots.get(s, []) if gm in _VOICE1]
            if not hits:
                measure.append(_rest_el(voice=1))
            else:
                for i, (gm, ghost) in enumerate(hits):
                    step, octave, stem, head = _PERC[gm]
                    measure.append(_note_el(
                        step, octave, stem, head,
                        voice=1, is_chord=(i > 0), ghost=ghost,
                    ))

        # ── Backup to start of measure ───────────────────────────────────────
        measure.append(_backup_el(slots_per_measure))

        # ── Voice 2 (stems-down: BD) ─────────────────────────────────────────
        for s in range(slots_per_measure):
            hits = [(gm, gh) for gm, gh in slots.get(s, []) if gm in _VOICE2]
            if not hits:
                measure.append(_rest_el(voice=2))
            else:
                for i, (gm, ghost) in enumerate(hits):
                    step, octave, stem, head = _PERC[gm]
                    measure.append(_note_el(
                        step, octave, stem, head,
                        voice=2, is_chord=(i > 0), ghost=ghost,
                    ))

    # ── Serialise ────────────────────────────────────────────────────────────
    ET.indent(root, space='  ')
    xml_bytes = ET.tostring(root, encoding='unicode', xml_declaration=False)

    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC\n'
        '  "-//Recordare//DTD MusicXML 3.1 Partwise//EN"\n'
        '  "http://www.musicxml.org/dtds/partwise.dtd">\n'
    )
    return header + xml_bytes
