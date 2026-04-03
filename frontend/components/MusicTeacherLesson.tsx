'use client';

import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import CatLiLogo from '@/components/CatLiLogo';

interface MusicTeacherLessonProps {
  exercises: string;
}

// Per-section colour tokens (pastel palette)
const HEADER_STYLES: Record<string, { border: string; bg: string; text: string; dot: string }> = {
  'overall':      { border: '#C4B5FD', bg: '#F3F0FF', text: '#4C3D9E', dot: '#C4B5FD' },
  'rudiment':     { border: '#86EFAC', bg: '#F0FDF4', text: '#166534', dot: '#86EFAC' },
  'ghost':        { border: '#F9A8D4', bg: '#FDF2F8', text: '#9D174D', dot: '#F9A8D4' },
  'groove':       { border: '#FFD0A5', bg: '#FFFBF5', text: '#92400E', dot: '#FFD0A5' },
  'coordination': { border: '#FFD0A5', bg: '#FFFBF5', text: '#92400E', dot: '#FFD0A5' },
};

function getHeaderStyle(text: string) {
  const lower = text.toLowerCase();
  for (const [key, style] of Object.entries(HEADER_STYLES)) {
    if (lower.includes(key)) return style;
  }
  return { border: '#EAE6FC', bg: '#F8F7FF', text: '#2D2B4E', dot: '#C4B5FD' };
}

const components: Components = {
  // ── Section banners ──────────────────────────────────────────────────────
  h2: ({ children }) => {
    const text = String(children);
    const s = getHeaderStyle(text);
    return (
      <h2
        style={{ borderLeftColor: s.border, backgroundColor: s.bg, color: s.text }}
        className="mt-8 mb-4 px-5 py-3 rounded-2xl border-l-4 text-sm font-bold tracking-tight"
      >
        {children}
      </h2>
    );
  },

  // ── Sub-headers ──────────────────────────────────────────────────────────
  h3: ({ children }) => (
    <h3 className="mt-5 mb-2 text-xs font-bold text-catli-purple-dark uppercase tracking-widest">
      {children}
    </h3>
  ),

  // ── Body text ────────────────────────────────────────────────────────────
  p: ({ children }) => (
    <p className="mb-3 text-sm leading-relaxed text-catli-text">
      {children}
    </p>
  ),

  // ── Lists ────────────────────────────────────────────────────────────────
  ol: ({ children }) => (
    <ol className="mb-4 ml-5 list-decimal space-y-1.5 text-sm text-catli-text">
      {children}
    </ol>
  ),
  ul: ({ children }) => (
    <ul className="mb-4 ml-5 list-disc space-y-1.5 text-sm text-catli-text">
      {children}
    </ul>
  ),
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,

  // ── Inline code — sticking patterns, note names ──────────────────────────
  code: ({ children }) => (
    <code className="px-1.5 py-0.5 rounded-lg text-xs font-mono
                     bg-catli-purple-light text-catli-purple-dark">
      {children}
    </code>
  ),

  // ── Code blocks — ASCII notation exercises ───────────────────────────────
  pre: ({ children }) => (
    <pre className="mb-4 overflow-x-auto rounded-2xl text-xs font-mono p-5 leading-relaxed
                    bg-catli-text text-catli-purple-light">
      {children}
    </pre>
  ),

  // ── Bold ─────────────────────────────────────────────────────────────────
  strong: ({ children }) => (
    <strong className="font-bold text-catli-purple-dark">{children}</strong>
  ),

  // ── Horizontal rule as a soft divider ────────────────────────────────────
  hr: () => <hr className="my-6 border-catli-border" />,
};

export default function MusicTeacherLesson({ exercises }: MusicTeacherLessonProps) {
  return (
    <section
      className="rounded-3xl border-2 border-catli-border p-8
                 shadow-[0_8px_30px_0_rgba(196,181,253,0.2)]"
      style={{
        /* Subtle lined-notepad feel via repeating gradient */
        background: `
          repeating-linear-gradient(
            to bottom,
            transparent,
            transparent 31px,
            #EAE6FC 31px,
            #EAE6FC 32px
          ),
          #FFFFFF
        `,
      }}
    >
      {/* Card header */}
      <div className="flex items-center gap-3 mb-6 pb-4 border-b-2 border-catli-border">
        <CatLiLogo size={36} />
        <div>
          <h2 className="text-base font-extrabold text-catli-text leading-tight">
            Music Teacher Lesson
          </h2>
          <p className="text-xs text-catli-muted">powered by qwen2.5:14b</p>
        </div>
      </div>

      <ReactMarkdown components={components}>{exercises}</ReactMarkdown>
    </section>
  );
}
