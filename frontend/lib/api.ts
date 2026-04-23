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

export async function downloadCloneHero(
  jobId: string,
  onProgress?: (pct: number, message: string) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const url = apiUrl(`/api/notation/clonehero/${encodeURIComponent(jobId)}/stream`);
    const es = new EventSource(url);

    es.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);
        if (onProgress) onProgress(data.pct ?? 0, data.message ?? '');

        if (data.download_token) {
          es.close();
          // Fetch the built zip using the token
          const res = await fetch(
            apiUrl(`/api/notation/clonehero/download/${data.download_token}`),
          );
          if (!res.ok) throw new Error('Failed to download Clone Hero chart');
          const blob = await res.blob();
          const blobUrl = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = blobUrl;
          a.download = `${jobId}_clonehero.zip`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(blobUrl);
          resolve();
        }
      } catch (err) {
        es.close();
        reject(err);
      }
    };

    es.onerror = () => {
      es.close();
      reject(new Error('Failed to build Clone Hero chart'));
    };
  });
}
