"""
Optional FAISS-backed semantic recall utilities.

The memory system can enrich retrieval by embedding turns/artifacts and
searching them with FAISS. This module keeps that logic isolated so the
core SQLite flow still works when FAISS or sentence-transformers are
unavailable.

Current state: lightweight shim that reports availability and documents
the intended interface. Real indexing/search will be added once the demo
needs semantic recall.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

_faiss_available = False
_import_error: str | None = None

# Optional heavy deps: only mark as loadable if present and not explicitly disabled.
_DISABLE_EMBEDDINGS = os.getenv("DISABLE_EMBEDDINGS", "").lower() in {"1", "true", "yes"}

def _probe_optional_deps() -> Tuple[bool, str | None]:
    if _DISABLE_EMBEDDINGS:
        return False, "disabled via DISABLE_EMBEDDINGS env var"
    faiss_spec = importlib.util.find_spec("faiss")
    st_spec = importlib.util.find_spec("sentence_transformers")
    if not faiss_spec or not st_spec:
        missing = "faiss" if not faiss_spec else "sentence_transformers"
        return False, f"missing optional dependency: {missing}"
    return True, None

_faiss_available, _import_error = _probe_optional_deps()


@dataclass
class EmbeddingHit:
    item_type: str
    item_id: int
    score: float


class EmbeddingIndex:
    """
    Minimal placeholder for FAISS-backed search.

    Usage intent:
    - Load a sentence-transformers model.
    - Add vectors to a FAISS index persisted at `index_path`.
    - Provide `search` to return ranked hits for a query string.

    Supported item types (text is flattened via `prepare_text_for_embedding`):
    - resume (pulled from a file path in env, e.g., RESUME_PATH)
    - job_description
    - company_research
    - cover_letter
    - email
    - recruiter_profile
    - user_preferences
    - conversation_turn
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str | Path = "faiss.index") -> None:
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.available = _faiss_available
        self._unavailable_reason = _import_error

        # Deliberately avoid loading heavy deps when unavailable.
        self._model = None
        self._index = None
        self._faiss = None

    def is_available(self) -> bool:
        """Return True if FAISS + sentence-transformers are loadable (without forcing model load)."""
        return self.available

    def _ensure_loaded(self) -> bool:
        """
        Lazily import heavy deps only when needed. Returns True on success, False otherwise.
        Does not raise to keep the shim non-fatal.
        """
        if not self.available:
            return False
        if self._model and self._index is not None and self._faiss:
            return True
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._faiss = faiss
            self._model = SentenceTransformer(self.model_name)
            # Index creation would go here in a full implementation.
            self._index = None
            return True
        except Exception as exc:  # noqa: BLE001
            self.available = False
            self._unavailable_reason = str(exc)
            return False

    def add(self, item_type: str, item_id: int, text: str) -> None:
        """
        Placeholder add. In a full build this would:
        - Encode text with the transformer model.
        - Add the vector to the FAISS index and persist to disk.
        """
        if not self._ensure_loaded():
            return
        # Real implementation to follow in semantic recall milestone.

    def search(self, query: str, top_k: int = 5) -> List[EmbeddingHit]:
        """
        Placeholder search. Returns an empty list when unavailable.

        Future behavior:
        - Encode the query string.
        - Retrieve nearest neighbors from the FAISS index.
        - Map neighbors back to (item_type, item_id) via embeddings_map.
        """
        if not self._ensure_loaded():
            return []
        # Real implementation to follow in semantic recall milestone.
        return []

    def debug_reason_unavailable(self) -> str | None:
        """Return the import error if embeddings are unavailable."""
        return self._unavailable_reason


SUPPORTED_ITEM_TYPES: Sequence[str] = (
    "resume",
    "job_description",
    "company_research",
    "cover_letter",
    "email",
    "recruiter_profile",
    "user_preferences",
    "conversation_turn",
)

# Import shared helpers from utils. Relative imports can be brittle when this
# module is executed directly, so fall back to loading by path.
try:
    from .utils import load_resume_from_env, prepare_text_for_embedding  # type: ignore
except Exception:
    utils_path = Path(__file__).parent / "utils.py"
    spec = importlib.util.spec_from_file_location("embedding_utils", utils_path)
    if spec and spec.loader:
        utils_module = importlib.util.module_from_spec(spec)
        sys.modules["embedding_utils"] = utils_module
        spec.loader.exec_module(utils_module)  # type: ignore
        load_resume_from_env = utils_module.load_resume_from_env  # type: ignore
        prepare_text_for_embedding = utils_module.prepare_text_for_embedding  # type: ignore
    else:
        raise


__all__: Sequence[str] = [
    "EmbeddingIndex",
    "EmbeddingHit",
    "SUPPORTED_ITEM_TYPES",
    "load_resume_from_env",
    "prepare_text_for_embedding",
]
