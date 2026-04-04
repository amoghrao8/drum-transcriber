'use client';

import { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabase';

// ---------------------------------------------------------------------------
// Types — mirror the MIDI-first backend schema
// ---------------------------------------------------------------------------

export interface DrumEvent {
  note:     number;   // GM MIDI drum note (36=kick, 38=snare, 42=hh, etc.)
  time:     number;   // seconds (quantised to 16th-note grid)
  velocity: number;   // 0-127
  type:     string;   // "kick" | "snare" | "hihat_closed" | "hihat_open" | "crash" | "ride" | "china"
  ghost:    boolean;  // velocity < 30th percentile of snare hits
  duration: number;   // seconds
}

export interface TrackMetadata {
  bpm:               number;
  time_signature:    string;   // e.g. "4/4"
  beats_per_bar:     number;
  beat_unit:         number;
  duration:          number;   // total seconds
  event_count:       number;
  grid_phase?:        number;  // seconds — quantization grid origin offset from t=0
  grid_subdivisions?: number;  // subdivisions per beat (4 = 16th notes)
}

export interface Transcription {
  id:          string;
  job_id:      string;
  youtube_url: string | null;
  event_count: number;
  events:      DrumEvent[];
  metadata:    TrackMetadata | null;
  exercises:   string | null;
  created_at:  string;
}

interface UseTranscriptionOptions {
  youtubeUrl?: string;
  jobId?:      string;
}

function extractVideoId(url: string): string | null {
  const match = url.match(
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})/
  );
  return match ? match[1] : null;
}

export function useTranscription({ youtubeUrl, jobId }: UseTranscriptionOptions) {
  const [data,    setData]    = useState<Transcription | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  useEffect(() => {
    if (!youtubeUrl && !jobId) return;

    let cancelled = false;

    const run = async () => {
      setLoading(true);
      setError(null);
      setData(null);

      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let query: any = supabase
          .from('drum_transcriptions')
          .select('id, job_id, youtube_url, event_count, events, metadata, exercises, created_at')
          .order('created_at', { ascending: false })
          .limit(1);

        if (jobId) {
          query = query.eq('job_id', jobId);
        } else if (youtubeUrl) {
          const videoId = extractVideoId(youtubeUrl);
          query = videoId
            ? query.ilike('youtube_url', `%${videoId}%`)
            : query.eq('youtube_url', youtubeUrl);
        }

        const { data: rows, error: sbError } = await query;

        if (cancelled) return;
        if (sbError) throw new Error(sbError.message);
        if (!rows || rows.length === 0) {
          throw new Error(
            'No transcription found. Make sure the analysis pipeline has been run for this URL.'
          );
        }

        setData(rows[0] as Transcription);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    run();
    return () => { cancelled = true; };
  }, [youtubeUrl, jobId]);

  return { data, loading, error };
}
