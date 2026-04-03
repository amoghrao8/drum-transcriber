'use client';

import { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabase';

export interface DrumEvent {
  time: number;
  type: 'kick' | 'snare' | 'snare_ghost' | 'hat';
  velocity: number;
}

export interface Transcription {
  id: string;
  job_id: string;
  youtube_url: string | null;
  event_count: number;
  events: DrumEvent[];
  exercises: string | null;
  created_at: string;
}

interface UseTranscriptionOptions {
  youtubeUrl?: string;
  jobId?: string;
}

function extractVideoId(url: string): string | null {
  const match = url.match(
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})/
  );
  return match ? match[1] : null;
}

export function useTranscription({ youtubeUrl, jobId }: UseTranscriptionOptions) {
  const [data, setData] = useState<Transcription | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
          .select('*')
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
            "No transcription found. Make sure the analysis pipeline has been run for this URL."
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
