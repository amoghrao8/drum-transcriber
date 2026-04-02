import os
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

FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"
YTDLP_PATH = "/opt/homebrew/bin/yt-dlp"

# Use MPS on Apple Silicon if available, fall back to CPU
def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
