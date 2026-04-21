'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { apiUrl } from '@/lib/api';

export type PipelineStatus = 'idle' | 'starting' | 'running' | 'complete' | 'error';

export interface PipelineState {
  status: PipelineStatus;
  step: number;       // 1–4, 0 = not started
  stepName: string;
  pct: number;        // 0–100
  message: string;
  error: string | null;
  dbJobId: string | null;
}

const IDLE: PipelineState = {
  status: 'idle', step: 0, stepName: '', pct: 0, message: '', error: null, dbJobId: null,
};

export interface PipelineOverrides {
  override_bpm?: number | null;
  override_beats_per_bar?: number | null;
  override_beat_unit?: number | null;
}

export function usePipeline() {
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [state, setState] = useState<PipelineState>(IDLE);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Poll status endpoint every 5 seconds ─────────────────────────────────
  const poll = useCallback(async (id: string) => {
    try {
      const res = await fetch(apiUrl(`/api/pipeline/${id}`));
      if (!res.ok) return;
      const data = await res.json();

      setState({
        status:   data.status,
        step:     data.step,
        stepName: data.step_name,
        pct:      data.pct,
        message:  data.message,
        error:    data.error ?? null,
        dbJobId:  data.db_job_id ?? null,
      });

      // Stop polling once terminal state reached
      if (data.status === 'complete' || data.status === 'error') {
        if (intervalRef.current) clearInterval(intervalRef.current);
      }
    } catch {
      // Network error — keep polling
    }
  }, []);

  useEffect(() => {
    if (!pipelineId) return;

    const id = pipelineId;
    // Defer first poll to avoid synchronous setState in effect body
    const timeout = setTimeout(() => poll(id), 0);
    intervalRef.current = setInterval(() => poll(id), 5000);

    return () => {
      clearTimeout(timeout);
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [pipelineId, poll]);

  // ── Public API ────────────────────────────────────────────────────────────
  const start = useCallback(async (youtubeUrl: string, overrides?: PipelineOverrides) => {
    setState({ ...IDLE, status: 'starting', message: 'Connecting to backend...' });

    try {
      const res = await fetch(apiUrl('/api/pipeline/start'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          youtube_url: youtubeUrl,
          ...overrides,
        }),
      });

      if (!res.ok) {
        const err = await res.text();
        setState({ ...IDLE, status: 'error', error: `Failed to start pipeline: ${err}` });
        return;
      }

      const { pipeline_id } = await res.json();
      setPipelineId(pipeline_id);
    } catch {
      setState({
        ...IDLE, status: 'error',
        error: 'Could not reach the backend. Is the FastAPI server running on port 8000?',
      });
    }
  }, []);

  const reset = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setPipelineId(null);
    setState(IDLE);
  }, []);

  return { state, start, reset };
}
