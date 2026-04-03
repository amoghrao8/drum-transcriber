'use client';

import { useEffect, useRef, useState } from 'react';
import type { DrumEvent, TrackMetadata } from '@/hooks/useTranscription';

// ---------------------------------------------------------------------------
// GM drum note → VexFlow staff position + notehead style
// Percussion clef convention (treble staff lines: E4 G4 B4 D5 F5)
//   Kick (36)        – F4, stem down, oval notehead
//   Snare (38)       – C5, stem up, oval notehead
//   HH Closed (42)   – F5, stem up, X notehead
//   HH Open (46)     – G5, stem up, X notehead
//   Crash (49)       – A5, stem up, X notehead
//   Ride (51)        – E5, stem up, X notehead
//   China (52)       – B5, stem up, X notehead
//   Clap (39)        – C5, stem up, X notehead
// ---------------------------------------------------------------------------
interface NoteSpec { key: string; noteType?: string; stemDir: -1 | 1 }

const GM: Record<number, NoteSpec> = {
  36: { key: 'f/4',                   stemDir: -1 },   // Kick
  38: { key: 'c/5',                   stemDir:  1 },   // Snare
  39: { key: 'c/5', noteType: 'x',   stemDir:  1 },   // Clap
  42: { key: 'f/5', noteType: 'x',   stemDir:  1 },   // HH Closed
  46: { key: 'g/5', noteType: 'x',   stemDir:  1 },   // HH Open
  49: { key: 'a/5', noteType: 'x',   stemDir:  1 },   // Crash
  51: { key: 'e/5', noteType: 'x',   stemDir:  1 },   // Ride
  52: { key: 'b/5', noteType: 'x',   stemDir:  1 },   // China
};

const NOTE_LABEL: Record<number, string> = {
  36: 'K', 38: 'S', 39: 'ClSt', 42: 'HH', 46: 'OH', 49: 'Cr', 51: 'Ri', 52: 'Tr',
};

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------
const MEASURES_PER_ROW = 4;
const STAVE_HEIGHT     = 80;    // px between stave Y positions
const ROW_GAP          = 40;    // extra px between rows
const TOP_MARGIN       = 20;
const LEFT_MARGIN      = 10;

// ---------------------------------------------------------------------------
// Helper: group events by (measure, 16th-note slot)
// ---------------------------------------------------------------------------
function groupEvents(
  events: DrumEvent[],
  bpm: number,
  beatsPerBar: number,
): Map<number, Map<number, DrumEvent[]>> {
  const sixteenth    = 60 / (bpm * 4);
  const measureSlots = beatsPerBar * 4;               // 16th-note slots per bar
  const measureDur   = sixteenth * measureSlots;

  // measureIdx → slotInMeasure → events[]
  const byMeasure = new Map<number, Map<number, DrumEvent[]>>();

  for (const ev of events) {
    const mIdx = Math.floor(ev.time / measureDur);
    const slot = Math.round((ev.time % measureDur) / sixteenth);
    const safeSlot = Math.min(slot, measureSlots - 1);

    if (!byMeasure.has(mIdx)) byMeasure.set(mIdx, new Map());
    const slotMap = byMeasure.get(mIdx)!;
    if (!slotMap.has(safeSlot)) slotMap.set(safeSlot, []);
    slotMap.get(safeSlot)!.push(ev);
  }
  return byMeasure;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
interface DrumNotationProps {
  events:   DrumEvent[];
  metadata: TrackMetadata | null;
}

export default function DrumNotation({ events, metadata }: DrumNotationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [rendered, setRendered] = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  const bpm          = metadata?.bpm          ?? 120;
  const beatsPerBar  = metadata?.beats_per_bar ?? 4;
  const beatUnit     = metadata?.beat_unit     ?? 4;
  const timeSig      = metadata?.time_signature ?? '4/4';
  const totalSeconds = metadata?.duration      ?? 0;

  useEffect(() => {
    if (!containerRef.current || !events.length) return;
    const container = containerRef.current;
    container.innerHTML = '';
    setRendered(false);
    setError(null);

    import('vexflow').then((VF) => {
      try {
        const { Renderer, Stave, StaveNote, Voice, Formatter, Beam } = VF;

        // ── Measure grouping ──────────────────────────────────────────────
        const byMeasure    = groupEvents(events, bpm, beatsPerBar);
        const totalMeasures = Math.min((byMeasure.size || 1) + 1, 200);
        const slotCount    = beatsPerBar * 4;   // 16th-note slots per bar

        const numRows    = Math.ceil(totalMeasures / MEASURES_PER_ROW);
        const totalHeight = TOP_MARGIN + numRows * (STAVE_HEIGHT + ROW_GAP) + 20;
        const width       = container.clientWidth || 900;
        const measureW    = Math.floor((width - LEFT_MARGIN * 2) / MEASURES_PER_ROW);

        const renderer = new Renderer(container, Renderer.Backends.SVG);
        renderer.resize(width, totalHeight);
        const ctx = renderer.getContext();
        ctx.setFont('Arial', 10);

        for (let mIdx = 0; mIdx < totalMeasures; mIdx++) {
          const col = mIdx % MEASURES_PER_ROW;
          const row = Math.floor(mIdx / MEASURES_PER_ROW);
          const sx  = LEFT_MARGIN + col * measureW;
          const sy  = TOP_MARGIN  + row * (STAVE_HEIGHT + ROW_GAP);

          // First measure of each row gets clef; very first also gets time sig
          const isFirstInRow  = col === 0;
          const staveDispW    = isFirstInRow ? measureW : measureW;
          const stave = new Stave(sx, sy, staveDispW - 4);

          if (isFirstInRow) stave.addClef('percussion');
          if (mIdx === 0)   stave.addTimeSignature(timeSig);
          stave.setContext(ctx).draw();

          const slotMap = byMeasure.get(mIdx) ?? new Map<number, DrumEvent[]>();

          // Build two voices: stems-down (kick) and stems-up (everything else)
          const kickNotes: InstanceType<typeof StaveNote>[] = [];
          const topNotes:  InstanceType<typeof StaveNote>[] = [];

          for (let s = 0; s < slotCount; s++) {
            const slotEvs = slotMap.get(s) ?? [];

            // ── Voice 2: kick ───────────────────────────────────────────
            const kicks = slotEvs.filter(e => e.note === 36);
            if (kicks.length) {
              const n = new StaveNote({ keys: ['f/4'], duration: '16', stemDirection: -1 });
              kickNotes.push(n);
            } else {
              kickNotes.push(new StaveNote({ keys: ['b/4'], duration: '16r', stemDirection: -1 }));
            }

            // ── Voice 1: snare + cymbals ────────────────────────────────
            const topEvs = slotEvs.filter(e => e.note !== 36);
            if (topEvs.length) {
              // Deduplicate keys (same GM note at same slot → one notehead)
              const seenKeys = new Set<string>();
              const keys: string[] = [];
              const noteTypes: (string | undefined)[] = [];
              const ghosts: boolean[] = [];

              for (const ev of topEvs) {
                const spec = GM[ev.note] ?? { key: 'c/5', stemDir: 1 as 1 };
                if (!seenKeys.has(spec.key)) {
                  seenKeys.add(spec.key);
                  keys.push(spec.key);
                  noteTypes.push(spec.noteType);
                  ghosts.push(ev.ghost ?? false);
                }
              }

              // VexFlow StaveNote uses the first noteType for all keys
              const hasX = noteTypes.some(t => t === 'x');
              const noteStruct: Record<string, unknown> = {
                keys,
                duration:      '16',
                stemDirection:  1,
              };
              if (hasX) noteStruct.noteType = 'x';

              const n = new StaveNote(noteStruct as Parameters<typeof StaveNote>[0]);

              // Style each notehead
              keys.forEach((_, ki) => {
                const ev = topEvs[ki];
                if (!ev) return;
                if (ev.ghost) {
                  // Ghost notes: muted purple
                  n.setKeyStyle(ki, { fillStyle: '#C4B5FD', strokeStyle: '#9D8FDB' });
                } else if (ev.note === 42 || ev.note === 46) {
                  // Hi-hat: catli purple
                  n.setKeyStyle(ki, { fillStyle: '#7C6FCD', strokeStyle: '#7C6FCD' });
                } else if (ev.note === 49) {
                  // Crash: warm orange
                  n.setKeyStyle(ki, { fillStyle: '#E89A50', strokeStyle: '#E89A50' });
                } else if (ev.note === 51) {
                  // Ride: gold
                  n.setKeyStyle(ki, { fillStyle: '#D4A017', strokeStyle: '#D4A017' });
                } else if (ev.note === 52) {
                  // China / Trash Stack: red-amber
                  n.setKeyStyle(ki, { fillStyle: '#DC2626', strokeStyle: '#DC2626' });
                } else if (ev.note === 39) {
                  // Clap Stack: teal
                  n.setKeyStyle(ki, { fillStyle: '#0891B2', strokeStyle: '#0891B2' });
                }
              });

              topNotes.push(n);
            } else {
              topNotes.push(new StaveNote({ keys: ['b/4'], duration: '16r', stemDirection: 1 }));
            }
          }

          // Format and draw
          try {
            const vKick = new Voice({ numBeats: beatsPerBar, beatValue: beatUnit });
            vKick.setStrict(false);
            vKick.addTickables(kickNotes);

            const vTop = new Voice({ numBeats: beatsPerBar, beatValue: beatUnit });
            vTop.setStrict(false);
            vTop.addTickables(topNotes);

            const fmtW = staveDispW - (isFirstInRow ? (mIdx === 0 ? 90 : 60) : 24);
            new Formatter().joinVoices([vKick, vTop]).format([vKick, vTop], Math.max(fmtW, 60));

            vKick.draw(ctx, stave);
            vTop.draw(ctx, stave);
          } catch {
            // Skip malformed measure silently
          }
        }

        setRendered(true);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events, bpm, beatsPerBar, beatUnit]);

  const ghostCount  = events.filter(e => e.ghost).length;
  const kickCount   = events.filter(e => e.note === 36).length;
  const snareCount  = events.filter(e => e.note === 38).length;
  const clapCount   = events.filter(e => e.note === 39).length;
  const hatCount    = events.filter(e => e.note === 42 || e.note === 46).length;
  const crashCount  = events.filter(e => e.note === 49).length;
  const rideCount   = events.filter(e => e.note === 51).length;
  const trashCount  = events.filter(e => e.note === 52).length;

  const totalMeasures = Math.min(
    Math.ceil(events.length > 0
      ? (Math.max(...events.map(e => e.time)) / (60 / bpm * beatsPerBar)) + 1
      : 1),
    200
  );

  return (
    <div className="bg-white rounded-3xl border-2 border-catli-border
                    shadow-[0_8px_30px_0_rgba(196,181,253,0.25)] overflow-hidden">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="px-6 pt-5 pb-4 border-b border-catli-border">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="font-bold text-catli-text text-sm">Drum Score</h3>
            <p className="text-xs text-catli-muted mt-0.5">
              {timeSig} &middot; {bpm} BPM &middot; {totalMeasures} measures
              &middot; {Math.round(totalSeconds)}s
            </p>
          </div>
          {/* Hit type pills */}
          <div className="flex flex-wrap gap-1.5 text-[10px]">
            {kickCount  > 0 && <span className="px-2 py-0.5 rounded-full bg-catli-bg text-catli-text font-mono">K {kickCount}</span>}
            {snareCount > 0 && <span className="px-2 py-0.5 rounded-full bg-catli-bg text-catli-text font-mono">S {snareCount}</span>}
            {ghostCount > 0 && <span className="px-2 py-0.5 rounded-full bg-[#EDE8FF] text-catli-purple-dark font-mono">g {ghostCount}</span>}
            {clapCount  > 0 && <span className="px-2 py-0.5 rounded-full bg-[#E0F2FE] text-[#0891B2] font-mono">ClSt {clapCount}</span>}
            {hatCount   > 0 && <span className="px-2 py-0.5 rounded-full bg-catli-purple-light text-catli-purple-dark font-mono">HH {hatCount}</span>}
            {crashCount > 0 && <span className="px-2 py-0.5 rounded-full bg-[#FFF3E0] text-catli-orange-dark font-mono">Cr {crashCount}</span>}
            {rideCount  > 0 && <span className="px-2 py-0.5 rounded-full bg-[#FEF9C3] text-[#92400E] font-mono">Ri {rideCount}</span>}
            {trashCount > 0 && <span className="px-2 py-0.5 rounded-full bg-[#FEE2E2] text-[#DC2626] font-mono">Tr {trashCount}</span>}
          </div>
        </div>

        {/* Colour legend */}
        <div className="flex flex-wrap gap-3 mt-2.5 text-[10px] text-catli-muted">
          <span><span className="inline-block w-2 h-2 rounded-full bg-catli-text mr-1" />Kick/Snare</span>
          <span><span className="inline-block w-2 h-2 rounded-full bg-[#7C6FCD] mr-1" />Hi-hat</span>
          <span><span className="inline-block w-2 h-2 rounded-full bg-[#E89A50] mr-1" />Crash</span>
          <span><span className="inline-block w-2 h-2 rounded-full bg-[#D4A017] mr-1" />Ride</span>
          <span><span className="inline-block w-2 h-2 rounded-full bg-[#DC2626] mr-1" />Trash</span>
          <span><span className="inline-block w-2 h-2 rounded-full bg-[#0891B2] mr-1" />Clap Stack</span>
          <span><span className="inline-block w-2 h-2 rounded-full bg-[#C4B5FD] mr-1" />Ghost</span>
          <span className="ml-auto">X = cymbal notehead</span>
        </div>
      </div>

      {/* ── Score canvas (scrollable) ───────────────────────────────────── */}
      <div className="overflow-y-auto" style={{ maxHeight: '60vh' }}>
        {error ? (
          <div className="p-6 text-xs text-red-600 font-mono">{error}</div>
        ) : (
          <div
            ref={containerRef}
            className="w-full bg-white px-2 py-3"
            style={{ minHeight: 200 }}
          />
        )}
      </div>

      {!rendered && !error && (
        <div className="px-6 py-3 text-xs text-catli-muted border-t border-catli-border">
          Rendering {totalMeasures} measures…
        </div>
      )}
    </div>
  );
}
