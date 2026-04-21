const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export const apiUrl = (path: string) => `${BASE}${path}`;

export async function downloadMidi(jobId: string): Promise<void> {
  const res = await fetch(apiUrl(`/api/notation/midi/${encodeURIComponent(jobId)}`));
  if (!res.ok) throw new Error('Failed to download MIDI file');
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${jobId}.mid`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
