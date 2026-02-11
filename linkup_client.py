from __future__ import annotations

from typing import Any, Dict, Optional
import os

# Auto-load .env when running scripts/tests directly (optional dependency)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


class LinkupClient:
    """
    Thin, stable wrapper around the Linkup SDK.
    Keeps the rest of the codebase decoupled from SDK changes.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("LINKUP_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing LINKUP_API_KEY environment variable. "
                "Add it to .env in repo root or export it in your shell."
            )

        try:
            from linkup import LinkupClient as _SDKClient
        except Exception as e:
            raise RuntimeError("Linkup SDK not installed. Run: pip install linkup-sdk") from e

        self._client = _SDKClient(api_key=self.api_key)

    def search(
        self,
        *,
        query: str,
        depth: str = "standard",             # "standard" | "deep"
        output_type: str = "searchResults",  # or "structured"
        schema: Optional[Any] = None,        # pydantic BaseModel class for your SDK
        max_results: int = 10,
        recency_days: Optional[int] = None,  # kept for compatibility; enforce via prompt
        include_images: bool = False,        # compatibility with older callers
    ) -> Dict[str, Any]:
        """
        Execute an agentic search via Linkup.

        NOTE:
        Some linkup-sdk versions do NOT accept `recency_days` or `include_images`.
        We keep them in the interface but do not pass them to the SDK.
        Enforce recency in the query string instead.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "depth": depth,
            "max_results": max_results,
        }

        _ = recency_days
        _ = include_images

        if output_type == "structured":
            if schema is None:
                raise ValueError("schema is required when output_type='structured'")
            payload["output_type"] = "structured"
            payload["structured_output_schema"] = schema
        else:
            payload["output_type"] = output_type

        resp = self._client.search(**payload)

        # Normalize response → dict
        if isinstance(resp, dict):
            return resp
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "dict"):
            return resp.dict()
        return {"raw": resp}


# Convenience functional wrapper (used by agents)
_client: Optional[LinkupClient] = None


def linkup_search(
    *,
    query: str,
    depth: str = "standard",
    output_type: str = "searchResults",
    schema: Optional[Any] = None,
    max_results: int = 10,
    recency_days: Optional[int] = None,
    include_images: bool = False,
) -> Dict[str, Any]:
    global _client
    if _client is None:
        _client = LinkupClient()

    return _client.search(
        query=query,
        depth=depth,
        output_type=output_type,
        schema=schema,
        max_results=max_results,
        recency_days=recency_days,
        include_images=include_images,
    )
