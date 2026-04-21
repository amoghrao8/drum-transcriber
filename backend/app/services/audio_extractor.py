import re
import uuid
import asyncio
import subprocess
from pathlib import Path

from app.core.config import AUDIO_OUTPUT_DIR, FFMPEG_PATH, YTDLP_PATH


def _is_valid_youtube_url(url: str) -> bool:
    pattern = r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}"
    return bool(re.match(pattern, url))


def _run_extraction_command(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "check": False,
    }
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(cmd, **kwargs)


async def extract_audio(youtube_url: str) -> dict:
    """
    Downloads audio from a YouTube URL and converts it to a 16-bit mono WAV
    at 44100 Hz. Returns metadata including the output file path.

    Raises ValueError for invalid URLs, RuntimeError for extraction failures.
    """
    if not _is_valid_youtube_url(youtube_url):
        raise ValueError(f"Invalid YouTube URL: {youtube_url}")

    job_id = uuid.uuid4().hex
    output_path = AUDIO_OUTPUT_DIR / f"{job_id}.wav"

    # yt-dlp: download best audio-only stream, pipe to ffmpeg for WAV conversion
    cmd = [
        YTDLP_PATH,
        "--no-playlist",
        "--quiet",
        "--extract-audio",
        "--audio-format", "wav",
        "--ffmpeg-location", FFMPEG_PATH,
        "--postprocessor-args", "ffmpeg:-ar 44100 -ac 1 -sample_fmt s16",
        "-o", str(output_path.with_suffix("")),  # yt-dlp appends .wav itself
        youtube_url,
    ]

    try:
        proc = await asyncio.to_thread(_run_extraction_command, cmd)
    except OSError as exc:
        raise RuntimeError(f"Failed to launch yt-dlp: {exc}") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {proc.returncode}): {proc.stderr.decode(errors='replace').strip()}"
        )

    # yt-dlp writes <job_id>.wav when the template has no extension
    if not output_path.exists():
        raise RuntimeError(f"Expected output file not found: {output_path}")

    return {
        "job_id": job_id,
        "file_path": str(output_path),
        "file_name": output_path.name,
        "size_bytes": output_path.stat().st_size,
    }
