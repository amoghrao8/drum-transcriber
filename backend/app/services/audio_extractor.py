import re
import asyncio
import subprocess
from pathlib import Path

from app.core.config import AUDIO_OUTPUT_DIR, FFMPEG_PATH, YTDLP_PATH, YTDLP_COOKIES_BROWSER


def _is_valid_youtube_url(url: str) -> bool:
    pattern = r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}"
    return bool(re.match(pattern, url))


def _sanitize_filename(title: str) -> str:
    """Sanitize a video title into a safe, human-readable filename."""
    # Remove characters that are problematic for file systems
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', title)
    # Replace whitespace runs with a single space
    safe = re.sub(r'\s+', ' ', safe).strip()
    # Truncate to a reasonable length
    if len(safe) > 120:
        safe = safe[:120].rstrip()
    return safe or "untitled"


def _run_extraction_command(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "check": False,
    }
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(cmd, **kwargs)


async def _fetch_video_title(youtube_url: str) -> str:
    """Use yt-dlp to fetch the video title."""
    cmd = [YTDLP_PATH, "--no-playlist", "--quiet", "--print", "title", youtube_url]
    try:
        proc = await asyncio.to_thread(_run_extraction_command, cmd)
    except OSError as exc:
        raise RuntimeError(f"Failed to launch yt-dlp: {exc}") from exc
    if proc.returncode != 0:
        return "untitled"
    return proc.stdout.decode(errors="replace").strip() or "untitled"


async def extract_audio(youtube_url: str) -> dict:
    """
    Downloads audio from a YouTube URL and converts it to a 16-bit mono WAV
    at 44100 Hz. Returns metadata including the output file path.

    Raises ValueError for invalid URLs, RuntimeError for extraction failures.
    """
    if not _is_valid_youtube_url(youtube_url):
        raise ValueError(f"Invalid YouTube URL: {youtube_url}")

    title = await _fetch_video_title(youtube_url)
    job_id = _sanitize_filename(title)

    # Deduplicate: if a file with this name already exists, append a counter
    output_path = AUDIO_OUTPUT_DIR / f"{job_id}.wav"
    counter = 1
    while output_path.exists():
        job_id = f"{_sanitize_filename(title)} ({counter})"
        output_path = AUDIO_OUTPUT_DIR / f"{job_id}.wav"
        counter += 1

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
