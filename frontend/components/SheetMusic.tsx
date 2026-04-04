'use client';

import { useEffect, useRef, useState } from 'react';
import { apiUrl } from '@/lib/api';

interface SheetMusicProps {
  jobId: string;
  title?: string;
}

export default function SheetMusic({ jobId, title }: SheetMusicProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<'loading' | 'rendered' | 'error'>('loading');
  const [error,  setError]  = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    let cancelled = false;
    setStatus('loading');
    setError(null);

    const render = async () => {
      try {
        // 1. Fetch MusicXML
        const res = await fetch(apiUrl(`/api/notation/musicxml/${jobId}`));
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail ?? `HTTP ${res.status}`);
        }
        const xml = await res.text();
        if (cancelled) return;

        // 2. Dynamically import OSMD
        const { OpenSheetMusicDisplay } = await import('opensheetmusicdisplay');
        if (cancelled || !containerRef.current) return;

        // 3. Clear and render — container must already be in the DOM with real dimensions
        containerRef.current.innerHTML = '';

        const osmd = new OpenSheetMusicDisplay(containerRef.current, {
          autoResize:            false,
          backend:               'svg',
          drawTitle:             false,
          drawSubtitle:          false,
          drawComposer:          false,
          drawLyricist:          false,
          drawPartNames:         false,
          drawPartAbbreviations: false,
        });

        await osmd.load(xml);
        osmd.render();

        if (!cancelled) setStatus('rendered');
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setStatus('error');
        }
      }
    };

    render();
    return () => { cancelled = true; };
  }, [jobId]);

  return (
    <div className="bg-white rounded-3xl border-2 border-catli-border
                    shadow-[0_8px_30px_0_rgba(196,181,253,0.25)] overflow-hidden">

      {/* Header */}
      <div className="px-6 pt-5 pb-3 border-b border-catli-border">
        <h3 className="font-bold text-catli-text text-sm">
          {title ?? 'Sheet Music'}
        </h3>
        <p className="text-xs text-catli-muted mt-0.5">
          Percussion notation &middot; 16th-note grid &middot; 2 voices
        </p>
      </div>

      {/* Score area — container is always in the DOM so OSMD gets real dimensions */}
      <div className="overflow-y-auto relative" style={{ maxHeight: '70vh' }}>

        {/* Loading overlay */}
        {status === 'loading' && (
          <div className="absolute inset-0 flex items-center justify-center gap-3
                          text-catli-muted text-sm bg-white z-10">
            <svg className="w-5 h-5 animate-spin text-catli-purple" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
            </svg>
            Generating score…
            <div style={{ height: 200 }} />
          </div>
        )}

        {/* Error */}
        {status === 'error' && (
          <div className="p-6 text-xs text-red-600 font-mono">{error}</div>
        )}

        {/* OSMD target — always mounted, always has width */}
        <div
          ref={containerRef}
          className="w-full bg-white px-4 py-4"
          style={{ minHeight: status === 'loading' ? 200 : undefined }}
        />
      </div>
    </div>
  );
}
