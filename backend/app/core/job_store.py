"""
In-memory pipeline job progress store.

Keyed by pipeline_id (a UUID hex string separate from the audio job_id).
Thread-safe via a simple lock — FastAPI runs async handlers on a single
thread per worker, but BackgroundTasks run in the thread pool.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineJob:
    pipeline_id: str
    youtube_url: str
    status: str = "pending"      # pending | running | complete | error
    step: int = 0                # 1-4
    step_name: str = "Queued"
    pct: int = 0                 # 0-100
    message: str = ""
    error: Optional[str] = None
    db_job_id: Optional[str] = None   # audio job_id written to Supabase


_store: dict[str, PipelineJob] = {}
_lock = threading.Lock()


def create_job(pipeline_id: str, youtube_url: str) -> PipelineJob:
    job = PipelineJob(pipeline_id=pipeline_id, youtube_url=youtube_url)
    with _lock:
        _store[pipeline_id] = job
    return job


def update_job(pipeline_id: str, **kwargs) -> None:
    with _lock:
        job = _store.get(pipeline_id)
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)


def get_job(pipeline_id: str) -> Optional[PipelineJob]:
    return _store.get(pipeline_id)
