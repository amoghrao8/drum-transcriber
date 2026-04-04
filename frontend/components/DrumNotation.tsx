'use client';

import type { DrumEvent, TrackMetadata } from '@/hooks/useTranscription';

// ---------------------------------------------------------------------------
// Instrument rows — top to bottom, matching ADTOF repo style
// ---------------------------------------------------------------------------
const ROWS = [
  { label: 'CY+RD', notes: new Set([49, 51]),     symbol: 'star', color: '#E89A50' },
  { label: 'HH',    notes: new Set([42, 46]),     symbol: 'x',    color: '#7C6FCD' },
  { label: 'TT',    notes: new Set([45, 47, 48]), symbol: 'dot',  color: '#16A34A' },
  { label: 'SD',    notes: new Set([38, 40]),     symbol: 'dot',  color: '#374151' },
  { label: 'BD',    notes: new Set([35, 36]),     symbol: 'dot',  color: '#1D4ED8' },
] as const;

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------
const ROW_H      = 36;   // px per instrument lane
const LABEL_W    = 54;   // px reserved for row labels on the left
const TOP_PAD    = 8;
const BOTTOM_PAD = 28;   // space for time axis
const PX_PER_SEC = 80;   // horizontal scale

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
interface DrumNotationProps {
  events:   DrumEvent[];
  metadata: TrackMetadata | null;
}

export default function DrumNotation({ events, metadata }: DrumNotationProps) {
  const totalDuration = metadata?.duration
    ?? (events.length ? Math.max(...events.map(e => e.time)) + 2 : 60);
  const bpm     = metadata?.bpm           ?? 120;
  const timeSig = metadata?.time_signature ?? '4/4';

  const svgW = Math.ceil(totalDuration * PX_PER_SEC) + LABEL_W;
  const svgH = ROWS.length * ROW_H + TOP_PAD + BOTTOM_PAD;

  // note → row index lookup
  const noteToRow = new Map<number, number>();
  ROWS.forEach((row, i) => row.notes.forEach(n => noteToRow.set(n, i)));

  // Time axis ticks
  const tickStep = totalDuration > 120 ? 5 : 1;
  const ticks: number[] = [];
  for (let t = 0; t <= totalDuration + tickStep / 2; t += tickStep) ticks.push(t);

  return (
    <div className="bg-white rounded-3xl border-2 border-catli-border
                    shadow-[0_8px_30px_0_rgba(196,181,253,0.25)] overflow-hidden">

      {/* Header */}
      <div className="px-6 pt-5 pb-3 border-b border-catli-border">
        <h3 className="font-bold text-catli-text text-sm">Drum Transcription</h3>
        <p className="text-xs text-catli-muted mt-0.5">
          {timeSig} &middot; {bpm} BPM &middot; {Math.round(totalDuration)}s
          &middot; {events.length} events
        </p>
      </div>

      {/* Scrollable plot */}
      <div className="overflow-x-auto">
        <svg
          width={svgW}
          height={svgH}
          style={{ display: 'block', fontFamily: 'monospace' }}
        >
          {/* Row bands + labels */}
          {ROWS.map((row, i) => {
            const y = TOP_PAD + i * ROW_H;
            return (
              <g key={row.label}>
                <rect
                  x={0} y={y} width={svgW} height={ROW_H}
                  fill={i % 2 === 0 ? '#FAFAFA' : '#FFFFFF'}
                />
                <line
                  x1={LABEL_W} y1={y + ROW_H}
                  x2={svgW}    y2={y + ROW_H}
                  stroke="#E5E7EB" strokeWidth={0.5}
                />
                <text
                  x={LABEL_W - 6} y={y + ROW_H / 2 + 4}
                  textAnchor="end" fontSize={10} fill="#6B7280"
                >
                  {row.label}
                </text>
              </g>
            );
          })}

          {/* Vertical time grid + axis */}
          {ticks.map(t => {
            const x = LABEL_W + t * PX_PER_SEC;
            return (
              <g key={t}>
                <line
                  x1={x} y1={TOP_PAD}
                  x2={x} y2={TOP_PAD + ROWS.length * ROW_H}
                  stroke="#E5E7EB" strokeWidth={0.5} strokeDasharray="2,4"
                />
                <line
                  x1={x} y1={TOP_PAD + ROWS.length * ROW_H}
                  x2={x} y2={TOP_PAD + ROWS.length * ROW_H + 5}
                  stroke="#9CA3AF" strokeWidth={1}
                />
                <text
                  x={x} y={TOP_PAD + ROWS.length * ROW_H + 18}
                  textAnchor="middle" fontSize={9} fill="#6B7280"
                >
                  {t.toFixed(0)}
                </text>
              </g>
            );
          })}

          {/* Events */}
          {events.map((ev, idx) => {
            const rowIdx = noteToRow.get(ev.note);
            if (rowIdx === undefined) return null;

            const row = ROWS[rowIdx];
            const cx  = LABEL_W + ev.time * PX_PER_SEC;
            const cy  = TOP_PAD + rowIdx * ROW_H + ROW_H / 2;

            if (row.symbol === 'x') {
              const s = 4.5;
              return (
                <g key={idx}>
                  <line x1={cx - s} y1={cy - s} x2={cx + s} y2={cy + s}
                    stroke={row.color} strokeWidth={1.5} />
                  <line x1={cx + s} y1={cy - s} x2={cx - s} y2={cy + s}
                    stroke={row.color} strokeWidth={1.5} />
                </g>
              );
            }

            if (row.symbol === 'star') {
              const s = 5;
              // 5-point star as a polygon
              const pts = Array.from({ length: 5 }, (_, k) => {
                const outer = ((k * 72 - 90) * Math.PI) / 180;
                const inner = ((k * 72 - 90 + 36) * Math.PI) / 180;
                return [
                  `${cx + s * Math.cos(outer)},${cy + s * Math.sin(outer)}`,
                  `${cx + (s * 0.4) * Math.cos(inner)},${cy + (s * 0.4) * Math.sin(inner)}`,
                ].join(' ');
              }).join(' ');
              return (
                <polygon key={idx} points={pts} fill={row.color} />
              );
            }

            // dot
            const r = ev.ghost ? 3 : 4.5;
            return (
              <circle key={idx} cx={cx} cy={cy} r={r}
                fill={ev.ghost ? '#C4B5FD' : row.color}
                opacity={ev.ghost ? 0.6 : 1}
              />
            );
          })}
        </svg>
      </div>

      {/* Legend */}
      <div className="px-6 py-2 border-t border-catli-border flex flex-wrap gap-4 text-[10px] text-catli-muted">
        <span><span className="inline-block w-2 h-2 rounded-full bg-[#1D4ED8] mr-1" />BD</span>
        <span><span className="inline-block w-2 h-2 rounded-full bg-[#374151] mr-1" />SD</span>
        <span><span className="inline-block w-2 h-2 rounded-full bg-[#16A34A] mr-1" />TT</span>
        <span><span className="inline-block w-2 h-2 rounded-full bg-[#7C6FCD] mr-1" />HH</span>
        <span><span className="inline-block w-2 h-2 rounded-full bg-[#E89A50] mr-1" />CY+RD</span>
        <span><span className="inline-block w-2 h-2 rounded-full bg-[#C4B5FD] mr-1" />ghost</span>
      </div>
    </div>
  );
}
