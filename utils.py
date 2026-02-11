"""
Shared helpers for loading user assets and normalizing text content.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


def load_resume_from_env(env_var: str = "RESUME_PATH") -> str | None:
    """
    Read the user's resume text from a path stored in an environment variable.
    Returns None if the env var is unset or the file cannot be read.
    """
    resume_path = os.getenv(env_var)
    if not resume_path:
        return None
    try:
        return Path(resume_path).read_text(encoding="utf-8")
    except OSError:
        return None


def normalize_content(content: str | Mapping) -> str:
    """
    Normalize various content types (dict or string) into a text blob for
    consistent downstream use (logging, storage, summarization, etc.).
    """
    if isinstance(content, Mapping):
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return str(content)


__all__ = ["load_resume_from_env", "normalize_content"]
