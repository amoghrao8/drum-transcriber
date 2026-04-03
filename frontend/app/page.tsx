'use client';

import { useState } from 'react';
import { Search, Loader2, AlertCircle } from 'lucide-react';
import { useTranscription } from '@/hooks/useTranscription';
import DrumNotation from '@/components/DrumNotation';
import MusicTeacherLesson from '@/components/MusicTeacherLesson';
import CatLiLogo from '@/components/CatLiLogo';

export default function Home() {
  const [inputValue, setInputValue] = useState('');
  const [submittedUrl, setSubmittedUrl] = useState('');

  const { data, loading, error } = useTranscription({ youtubeUrl: submittedUrl });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = inputValue.trim();
    if (!trimmed) return;
    setSubmittedUrl(trimmed);
  };

  return (
    <div className="min-h-screen bg-catli-bg font-sans">

      {/* ── Nav ─────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur-sm border-b border-catli-border">
        <div className="max-w-4xl mx-auto px-6 py-3 flex items-center gap-3">
          <CatLiLogo size={36} />
          <span className="font-extrabold text-xl text-catli-text tracking-tight">
            Cat<span className="text-catli-purple-dark">.li</span>
          </span>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-14 space-y-10">

        {/* ── Hero ────────────────────────────────────────────────────────── */}
        <section className="text-center space-y-4">
          <div className="flex justify-center mb-2">
            <CatLiLogo size={80} />
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-catli-text">
            Analyse any drum track
          </h1>
          <p className="text-catli-muted text-base max-w-md mx-auto leading-relaxed">
            Paste a YouTube URL to get the notation, ghost-note detail, and a purr-fect AI lesson.
          </p>
        </section>

        {/* ── Input ───────────────────────────────────────────────────────── */}
        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-catli-muted pointer-events-none" />
            <input
              type="url"
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              className="w-full pl-11 pr-4 py-3.5 rounded-2xl border-2 border-catli-border
                         bg-white text-sm text-catli-text placeholder-catli-muted
                         focus:outline-none focus:border-catli-purple transition-colors"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !inputValue.trim()}
            className="px-6 py-3.5 rounded-2xl bg-catli-orange hover:bg-catli-orange-hover
                       disabled:opacity-50 text-catli-text text-sm font-bold
                       transition-all duration-150 hover:scale-105 active:scale-95
                       shadow-[0_4px_14px_0_rgba(255,208,165,0.6)] whitespace-nowrap"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" /> Loading…
              </span>
            ) : (
              'View Analysis'
            )}
          </button>
        </form>

        {/* ── Error ───────────────────────────────────────────────────────── */}
        {error && (
          <div className="flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <p>{error}</p>
          </div>
        )}

        {/* ── Results ─────────────────────────────────────────────────────── */}
        {data && (
          <div className="space-y-8">

            {/* Meta pills */}
            <div className="flex flex-wrap items-center gap-2 text-xs text-catli-muted">
              <span className="px-3 py-1.5 rounded-full bg-catli-purple-light text-catli-purple-dark font-mono font-medium">
                {data.job_id.slice(0, 12)}&hellip;
              </span>
              <span className="px-3 py-1.5 rounded-full bg-catli-purple-light text-catli-purple-dark font-medium">
                {data.event_count.toLocaleString()} events
              </span>
              <span className="px-3 py-1.5 rounded-full bg-catli-purple-light text-catli-purple-dark font-medium">
                {new Date(data.created_at).toLocaleDateString()}
              </span>
            </div>

            {/* Notation */}
            <DrumNotation events={data.events} />

            {/* Lesson or placeholder */}
            {data.exercises ? (
              <MusicTeacherLesson exercises={data.exercises} />
            ) : (
              <div className="rounded-3xl border-2 border-dashed border-catli-border bg-white p-10 text-center space-y-3">
                <CatLiLogo size={52} />
                <p className="text-sm font-semibold text-catli-text">No lesson yet!</p>
                <p className="text-xs text-catli-muted">
                  Run{' '}
                  <code className="font-mono bg-catli-purple-light text-catli-purple-dark px-1.5 py-0.5 rounded-lg">
                    python trigger_lesson.py
                  </code>{' '}
                  to generate one with Qwen&nbsp;2.5.
                </p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <footer className="text-center pb-10 text-xs text-catli-muted">
        Cat.li &mdash; drum analysis powered by{' '}
        <span className="text-catli-purple-dark font-medium">Qwen&nbsp;2.5</span> &amp;{' '}
        <span className="text-catli-orange-dark font-medium">VexFlow</span>
      </footer>
    </div>
  );
}
