'use client';

import { CheckCircle2, Circle, Loader2, XCircle, RefreshCw, Copy, Check } from 'lucide-react';
import { useState } from 'react';
import type { PipelineState } from '@/hooks/usePipeline';
import CatLiLogo from '@/components/CatLiLogo';

interface Step {
  n: number;
  label: string;
  tech: string;
}

const STEPS: Step[] = [
  { n: 1, label: 'Downloading audio',    tech: 'yt-dlp + ffmpeg → 44.1 kHz WAV' },
  { n: 2, label: 'Isolating drum track', tech: 'Meta Demucs htdemucs model' },
  { n: 3, label: 'Transcribing hits',    tech: 'FFT onset detection + frequency classification' },
  { n: 4, label: 'Generating AI lesson', tech: 'Qwen 2.5 14B via Ollama' },
];

interface PipelineProgressProps {
  state: PipelineState;
  youtubeUrl: string;
  onReset: () => void;
}

function StepIcon({ status }: { status: 'done' | 'running' | 'waiting' | 'error' }) {
  if (status === 'done')
    return <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0" />;
  if (status === 'running')
    return <Loader2 className="w-5 h-5 text-catli-orange-dark shrink-0 animate-spin" />;
  if (status === 'error')
    return <XCircle className="w-5 h-5 text-red-400 shrink-0" />;
  return <Circle className="w-5 h-5 text-catli-border shrink-0" />;
}

function stepStatus(
  stepN: number,
  currentStep: number,
  pipelineStatus: PipelineState['status']
): 'done' | 'running' | 'waiting' | 'error' {
  if (pipelineStatus === 'error' && stepN === currentStep) return 'error';
  if (stepN < currentStep) return 'done';
  if (stepN === currentStep && pipelineStatus === 'running') return 'running';
  return 'waiting';
}

// Shorten URL for display
function shortUrl(url: string): string {
  try {
    const u = new URL(url);
    return u.hostname + u.pathname + (u.search.length > 20 ? u.search.slice(0, 20) + '…' : u.search);
  } catch {
    return url.slice(0, 50);
  }
}

export default function PipelineProgress({ state, youtubeUrl, onReset }: PipelineProgressProps) {
  const isError    = state.status === 'error';
  const isComplete = state.status === 'complete';
  const [copied, setCopied] = useState(false);

  const copyError = () => {
    navigator.clipboard.writeText(state.error ?? '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-white rounded-3xl border-2 border-catli-border p-8
                    shadow-[0_8px_30px_0_rgba(196,181,253,0.2)] space-y-6">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-start gap-4">
        <div className="shrink-0">
          <CatLiLogo size={44} />
        </div>
        <div className="min-w-0">
          <h3 className="font-extrabold text-catli-text text-base leading-snug">
            {isError    ? 'Something went wrong' :
             isComplete ? 'Analysis complete!' :
                          'Processing your track…'}
          </h3>
          <p className="text-xs text-catli-muted mt-0.5 truncate">{shortUrl(youtubeUrl)}</p>
          {!isError && !isComplete && (
            <p className="text-xs text-catli-muted mt-1">
              Status updates every 5 seconds
            </p>
          )}
        </div>
        {(isError || isComplete) && (
          <button
            onClick={onReset}
            className="ml-auto shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-xl
                       bg-catli-purple-light hover:bg-catli-purple text-catli-purple-dark
                       text-xs font-semibold transition-all duration-150 hover:scale-105 active:scale-95"
          >
            <RefreshCw size={12} /> Reset
          </button>
        )}
      </div>

      {/* ── Steps list ─────────────────────────────────────────────────── */}
      <div className="space-y-3">
        {STEPS.map((step) => {
          const status = stepStatus(step.n, state.step, state.status);
          const isRunning = status === 'running';
          return (
            <div
              key={step.n}
              className={`flex items-center gap-3 rounded-2xl px-4 py-3 transition-all duration-300
                ${isRunning
                  ? 'bg-catli-orange/20 border-2 border-catli-orange'
                  : status === 'done'
                  ? 'bg-emerald-50 border-2 border-emerald-100'
                  : status === 'error'
                  ? 'bg-red-50 border-2 border-red-200'
                  : 'bg-catli-bg border-2 border-catli-border opacity-60'
                }`}
            >
              <StepIcon status={status} />
              <div className="flex-1 min-w-0">
                <p className={`text-sm font-semibold leading-snug
                  ${status === 'done'    ? 'text-emerald-700' :
                    status === 'running' ? 'text-catli-text'  :
                    status === 'error'   ? 'text-red-700'     :
                                          'text-catli-muted'}`}>
                  {step.label}
                </p>
                <p className="text-xs text-catli-muted mt-0.5 truncate">{step.tech}</p>
              </div>
              <span className={`text-xs font-medium shrink-0 px-2 py-0.5 rounded-full
                ${status === 'done'    ? 'bg-emerald-100 text-emerald-700' :
                  status === 'running' ? 'bg-catli-orange text-catli-text' :
                  status === 'error'   ? 'bg-red-100 text-red-700'         :
                                        'bg-catli-border text-catli-muted'}`}>
                {status === 'done'    ? 'Done'    :
                 status === 'running' ? 'Running' :
                 status === 'error'   ? 'Error'   :
                                        'Waiting'}
              </span>
            </div>
          );
        })}
      </div>

      {/* ── Progress bar ───────────────────────────────────────────────── */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs font-semibold text-catli-muted">
          <span>{state.stepName || 'Starting…'}</span>
          <span>{state.pct}%</span>
        </div>
        <div className="h-3 rounded-full bg-catli-purple-light overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700 ease-out"
            style={{
              width: `${state.pct}%`,
              background: 'linear-gradient(to right, #C4B5FD, #FFD0A5)',
            }}
          />
        </div>
      </div>

      {/* ── Step detail message ─────────────────────────────────────────── */}
      {state.message && !isError && (
        <div className="rounded-2xl bg-catli-bg border border-catli-border px-4 py-3">
          <p className="text-xs text-catli-text leading-relaxed">{state.message}</p>
        </div>
      )}

      {/* ── Error detail ────────────────────────────────────────────────── */}
      {isError && state.error && (
        <div className="rounded-2xl bg-red-50 border border-red-200 px-4 py-3">
          <div className="flex items-center justify-between mb-1">
            <p className="text-xs font-semibold text-red-700">Error details</p>
            <button
              onClick={copyError}
              className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium
                         bg-red-100 hover:bg-red-200 text-red-700 transition-all duration-150
                         hover:scale-105 active:scale-95"
            >
              {copied ? <><Check size={11} /> Copied</> : <><Copy size={11} /> Copy</>}
            </button>
          </div>
          <p className="text-xs text-red-600 font-mono break-all">{state.error}</p>
        </div>
      )}
    </div>
  );
}
