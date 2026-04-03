'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Play, Pause, RotateCcw } from 'lucide-react';
import type { DrumEvent } from '@/hooks/useTranscription';

// ---------------------------------------------------------------------------
// BPM estimation (mirrors the Python logic in music_teacher.py)
// ---------------------------------------------------------------------------
function estimateBpm(events: DrumEvent[], maxTime = 60): number {
  const hatTimes = events
    .filter(e => e.time <= maxTime && e.type === 'hat')
    .map(e => e.time)
    .sort((a, b) => a - b);

  const times =
    hatTimes.length >= 8
      ? hatTimes
      : events
          .filter(e => e.time <= maxTime)
          .map(e => e.time)
          .sort((a, b) => a - b);

  if (times.length < 4) return 120;

  const iois: number[] = [];
  for (let i = 1; i < times.length; i++) {
    const d = times[i] - times[i - 1];
    if (d >= 0.04 && d <= 1.5) iois.push(d);
  }
  if (!iois.length) return 120;

  const counts = new Map<number, number>();
  for (const ioi of iois) {
    const bin = Math.round(ioi / 0.005);
    counts.set(bin, (counts.get(bin) ?? 0) + 1);
  }
  let maxCnt = 0, modalBin = 0;
  counts.forEach((cnt, bin) => { if (cnt > maxCnt) { maxCnt = cnt; modalBin = bin; } });

  const modalIoi = modalBin * 0.005;
  for (const divisor of [4, 2, 1]) {
    const bpm = 60 / (divisor * modalIoi);
    if (bpm >= 60 && bpm <= 240) return Math.round(bpm * 10) / 10;
  }
  return 120;
}

// ---------------------------------------------------------------------------
// Quantise events onto a 16th-note slot grid
// ---------------------------------------------------------------------------
function buildGrid(
  events: DrumEvent[],
  bpm: number,
  measures: number
): Map<number, Set<DrumEvent['type']>> {
  const sixteenth = 60 / (bpm * 4);
  const totalSlots = measures * 16;
  const grid = new Map<number, Set<DrumEvent['type']>>();
  for (let i = 0; i < totalSlots; i++) grid.set(i, new Set());

  for (const ev of events) {
    const slot = Math.round(ev.time / sixteenth);
    if (slot < totalSlots) grid.get(slot)!.add(ev.type);
  }
  return grid;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
interface DrumNotationProps {
  events: DrumEvent[];
}

interface StaveBounds {
  noteStartX: number;
  noteEndX: number;
  durationSec: number;
}

const MEASURES = 2;

export default function DrumNotation({ events }: DrumNotationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playheadPct, setPlayheadPct] = useState<number | null>(null);
  const [bounds, setBounds] = useState<StaveBounds | null>(null);
  const rafRef = useRef<number | null>(null);
  const startTsRef = useRef<number>(0);

  const bpm = estimateBpm(events);
  const sixteenth = 60 / (bpm * 4);
  const durationSec = MEASURES * 16 * sixteenth;

  // ── VexFlow render ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || !events.length) return;

    const container = containerRef.current;
    container.innerHTML = '';

    import('vexflow').then((VF) => {
      const { Renderer, Stave, StaveNote, Voice, Formatter } = VF;

      const width = container.clientWidth || 800;
      const height = 200;

      const renderer = new Renderer(container, Renderer.Backends.SVG);
      renderer.resize(width, height);
      const ctx = renderer.getContext();
      ctx.setFont('Arial', 10);

      const staveWidth = Math.floor((width - 20) / MEASURES);
      const grid = buildGrid(events, bpm, MEASURES);

      let firstNoteStartX = 0;

      for (let m = 0; m < MEASURES; m++) {
        const sx = 10 + m * staveWidth;
        const stave = new Stave(sx, 40, staveWidth);
        if (m === 0) {
          stave.addClef('percussion');
          stave.addTimeSignature('4/4');
        }
        stave.setContext(ctx).draw();

        const kickNotes: InstanceType<typeof StaveNote>[] = [];
        const topNotes:  InstanceType<typeof StaveNote>[] = [];

        for (let slot = 0; slot < 16; slot++) {
          const abs = m * 16 + slot;
          const hits = grid.get(abs) ?? new Set<DrumEvent['type']>();

          // Voice 1 — kick, stems down
          if (hits.has('kick')) {
            kickNotes.push(
              new StaveNote({ keys: ['f/4'], duration: '16', stemDirection: -1 })
            );
          } else {
            kickNotes.push(
              new StaveNote({ keys: ['b/4'], duration: '16r', stemDirection: -1 })
            );
          }

          // Voice 2 — snare / ghost / hat, stems up
          const topKeys: string[] = [];
          if (hits.has('hat'))         topKeys.push('g/5');
          if (hits.has('snare'))       topKeys.push('c/5');
          if (hits.has('snare_ghost') && !hits.has('snare')) topKeys.push('c/5');

          if (topKeys.length > 0) {
            const n = new StaveNote({ keys: topKeys, duration: '16', stemDirection: 1 });
            if (hits.has('hat')) {
              n.setKeyStyle(0, { fillStyle: '#7C6FCD', strokeStyle: '#7C6FCD' });
            }
            if (hits.has('snare_ghost') && !hits.has('snare')) {
              const ghostIdx = hits.has('hat') ? 1 : 0;
              n.setKeyStyle(ghostIdx, { fillStyle: '#C4B5FD', strokeStyle: '#C4B5FD' });
            }
            topNotes.push(n);
          } else {
            topNotes.push(
              new StaveNote({ keys: ['b/4'], duration: '16r', stemDirection: 1 })
            );
          }
        }

        const vKick = new Voice({ numBeats: 4, beatValue: 4 });
        vKick.addTickables(kickNotes);

        const vTop = new Voice({ numBeats: 4, beatValue: 4 });
        vTop.addTickables(topNotes);

        const formatWidth = staveWidth - (m === 0 ? 90 : 25);
        new Formatter().joinVoices([vKick, vTop]).format([vKick, vTop], formatWidth);

        vKick.draw(ctx, stave);
        vTop.draw(ctx, stave);

        if (m === 0) firstNoteStartX = stave.getNoteStartX();
      }

      setBounds({
        noteStartX:  firstNoteStartX,
        noteEndX:    10 + MEASURES * staveWidth,
        durationSec,
      });
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  // ── Playhead animation ──────────────────────────────────────────────────
  const animate = useCallback(
    (ts: number) => {
      if (!bounds) return;
      const elapsed  = (ts - startTsRef.current) / 1000;
      const progress = Math.min(elapsed / bounds.durationSec, 1);
      setPlayheadPct(progress * 100);
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      } else {
        setIsPlaying(false);
        setPlayheadPct(null);
      }
    },
    [bounds]
  );

  const handlePlay = () => {
    if (isPlaying) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      setIsPlaying(false);
      setPlayheadPct(null);
      return;
    }
    setIsPlaying(true);
    startTsRef.current = performance.now();
    rafRef.current = requestAnimationFrame(animate);
  };

  const handleReset = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    setIsPlaying(false);
    setPlayheadPct(null);
  };

  useEffect(() => () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); }, []);

  const playheadPx =
    bounds && playheadPct !== null
      ? bounds.noteStartX + (playheadPct / 100) * (bounds.noteEndX - bounds.noteStartX)
      : null;

  return (
    <div className="bg-white rounded-3xl border-2 border-catli-border p-6
                    shadow-[0_8px_30px_0_rgba(196,181,253,0.25)]">
      {/* Header row */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-bold text-catli-text text-sm">Drum Notation</h3>
          <p className="text-xs text-catli-muted mt-0.5">
            BPM &asymp;&nbsp;{bpm}&ensp;&middot;&ensp;First {MEASURES} measures&ensp;&middot;&ensp;
            K=Kick&nbsp;S=Snare&nbsp;g=Ghost&nbsp;H=Hi-hat
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handlePlay}
            disabled={!bounds}
            className="flex items-center gap-1.5 px-4 py-2 rounded-2xl text-sm font-bold
                       bg-catli-orange hover:bg-catli-orange-hover disabled:opacity-40
                       text-catli-text transition-all duration-150 hover:scale-105 active:scale-95
                       shadow-[0_4px_12px_0_rgba(255,208,165,0.5)]"
          >
            {isPlaying ? <Pause size={13} /> : <Play size={13} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-2 rounded-2xl text-sm
                       bg-catli-purple-light hover:bg-catli-purple text-catli-purple-dark
                       transition-all duration-150 hover:scale-105 active:scale-95"
          >
            <RotateCcw size={13} />
          </button>
        </div>
      </div>

      {/* Notation canvas — white so VexFlow ink stays legible */}
      <div className="relative overflow-hidden rounded-2xl bg-white border border-catli-border">
        <div ref={containerRef} className="w-full" style={{ minHeight: 200 }} />

        {/* Pastel orange playhead */}
        {playheadPx !== null && (
          <div
            className="absolute top-0 bottom-0 w-0.5 pointer-events-none"
            style={{
              left: playheadPx,
              background: 'linear-gradient(to bottom, #FFD0A5, #E89A50)',
              opacity: 0.9,
            }}
          />
        )}
      </div>
    </div>
  );
}
