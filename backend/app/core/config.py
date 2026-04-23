import os
import shutil
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

_BASE = Path(__file__).resolve().parents[2]

# Load .env from the monorepo root. This runs once at import time.
# Variables are only available to this Python process — never sent to the
# frontend or exposed outside the backend.
load_dotenv(_BASE.parent / ".env")  # .env lives at monorepo root, one above backend/

# ---------------------------------------------------------------------------
# Supabase — backend-only secrets
# ---------------------------------------------------------------------------
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# Directories for audio artifacts
AUDIO_OUTPUT_DIR = _BASE / "audio_files"
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

STEMS_OUTPUT_DIR = _BASE / "stems"
STEMS_OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# External binary resolution
# Checks the venv's Scripts/ dir first, then falls back to PATH.
# ---------------------------------------------------------------------------
def _resolve_binary(name: str) -> str:
    ext = ".exe" if sys.platform == "win32" else ""
    venv_bin = Path(sys.executable).parent / f"{name}{ext}"
    if venv_bin.exists():
        return str(venv_bin)
    on_path = shutil.which(name)
    if on_path:
        return on_path
    raise RuntimeError(
        f"Required binary '{name}' not found in venv ({venv_bin}) or system PATH. "
        f"Install it with: pip install {name}"
    )

FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"
YTDLP_PATH  = _resolve_binary("yt-dlp")

# Browser to read cookies from for yt-dlp (e.g. "chrome", "firefox", "edge").
# Set YTDLP_COOKIES_BROWSER env var to override. Empty string disables.
YTDLP_COOKIES_BROWSER: str = os.environ.get("YTDLP_COOKIES_BROWSER", "chrome")

# Use CUDA on NVIDIA GPU if available, fall back to CPU
def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
