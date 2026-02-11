# linkup_client.py
from __future__ import annotations

"""
Thin wrapper around LinkUp search + helper normalization utilities.

This file intentionally:
- Keeps a stable LinkupClient wrapper so the rest of the codebase is decoupled from SDK changes.
- Provides a convenience linkup_search() function used by agents.
- Preserves LinkupJobSearch + research_from_selected_jd() flow.
- Adds normalize_search_results_to_jobs() used by app.py to render job cards.
"""

import os
import re
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

# Auto-load .env when running scripts/tests directly
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --- Prefer the real SDK ---
# linkup-sdk exposes `linkup_sdk`, not `linkup` (which may be an unrelated package).
_SDK_IMPORT_ERROR: Exception | None = None
_SDKClient = None

try:
    from linkup_sdk import LinkupClient as _SDKClient  # type: ignore
except Exception as e1:
    _SDK_IMPORT_ERROR = e1
    try:
        # Fallback if repo previously used "from linkup import LinkupClient"
        from linkup import LinkupClient as _SDKClient  # type: ignore
        _SDK_IMPORT_ERROR = None
    except Exception as e2:
        _SDK_IMPORT_ERROR = e2
        _SDKClient = None

from company_research_agent import CompanyResearchAgent, JobPostingIntake


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


# -------------------------------------------------------------------
# Normalization helpers (for app.py job cards)
# -------------------------------------------------------------------

_LINE_KV_RE = re.compile(r"(?im)^\s*(company|employer|location|title|role)\s*:\s*(.+?)\s*$")


def normalize_search_results_to_jobs(
    response: Any,
    *,
    role: Optional[str] = None,
    company: Optional[str] = None,
    location: Optional[str] = None,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    """
    Convert LinkUp `searchResults` response into simple job "cards".

    We do best-effort extraction:
    - one card per result
    - try to infer title/company/location from the content (when available)
    - dedupe by URL
    - include bounded jd_text (content) to support downstream resume tailoring
    """
    results = None
    if isinstance(response, dict):
        results = response.get("results")
    if results is None:
        results = _safe_getattr(response, "results", None)

    if not isinstance(results, list):
        return []

    def extract_kv(content: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not content:
            return out
        for m in _LINE_KV_RE.finditer(content):
            k = (m.group(1) or "").strip().lower()
            v = (m.group(2) or "").strip()
            if k and v and k not in out:
                out[k] = v
        return out

    jobs: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for i, r in enumerate(results, start=1):
        if len(jobs) >= limit:
            break

        name = None
        url = None
        content = ""

        if isinstance(r, dict):
            name = (r.get("name") or r.get("title") or "").strip() or None
            url = (r.get("url") or "").strip() or None
            content = (r.get("content") or r.get("snippet") or "").strip()
        else:
            name = (_safe_getattr(r, "name", None) or "").strip() or None
            url = (_safe_getattr(r, "url", None) or "").strip() or None
            content = (_safe_getattr(r, "content", None) or "").strip() or ""

        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        kv = extract_kv(content)
        inferred_company = company or kv.get("company") or kv.get("employer")
        inferred_location = location or kv.get("location")
        inferred_title = kv.get("title") or kv.get("role") or name or (role or "NA")

        jd_text = content.strip()
        if jd_text:
            jd_text = jd_text[:8000]  # bound

        jobs.append(
            {
                "job_id": str(i),
                "title": inferred_title or "NA",
                "company": inferred_company or "NA",
                "location": inferred_location or "NA",
                "url": url or "NA",
                "snippet": (content[:400] + ("…" if len(content) > 400 else "")) if content else "NA",
                "jd_text": jd_text or "NA",
            }
        )

    return jobs


# -------------------------------------------------------------------
# Stable LinkupClient wrapper (preserved architecture)
# -------------------------------------------------------------------

class LinkupClient:
    """
    Thin, stable wrapper around the LinkUp SDK.

    Keeps the rest of the codebase decoupled from SDK changes.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("LINKUP_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing LINKUP_API_KEY environment variable. "
                "Add it to .env in repo root or export it in your shell."
            )

        if _SDKClient is None:
            hint = (
                f"Original import error: {type(_SDK_IMPORT_ERROR).__name__}: {_SDK_IMPORT_ERROR}"
                if _SDK_IMPORT_ERROR is not None
                else "SDK import failed for unknown reasons."
            )
            raise RuntimeError(
                "LinkUp SDK not available. Install the correct dependency:\n"
                "  pip install linkup-sdk\n\n"
                f"Python: {sys.executable}\n{hint}"
            )

        self._client = _SDKClient(api_key=self.api_key)

    def search(
        self,
        *,
        query: str,
        depth: str = "standard",             # "standard" | "deep"
        output_type: str = "searchResults",  # or "structured"
        schema: Optional[Any] = None,        # optional structured schema (if SDK supports)
        max_results: int = 10,
        recency_days: Optional[int] = None,  # kept for compatibility; enforce via prompt
        include_images: bool = False,        # kept for compatibility; not always supported by SDK
    ) -> Dict[str, Any]:
        """
        Execute an agentic search via LinkUp.

        NOTE:
        Some SDK versions do NOT accept `recency_days` or `include_images`.
        We keep them in the interface but do not pass them to the SDK.
        Enforce recency in the query string instead.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "depth": depth,
            "max_results": max_results,
            "output_type": output_type,
        }

        _ = recency_days
        _ = include_images

        # Only set structured args when requested
        if output_type == "structured":
            if schema is None:
                raise ValueError("schema is required when output_type='structured'")
            payload["structured_output_schema"] = schema

        try:
            resp = self._client.search(**payload)
        except Exception as e:
            tb = traceback.format_exc().rstrip()
            raise RuntimeError(
                f"LinkUp search failed: {type(e).__name__}: {e}\n\nPayload={payload}\n\nTraceback:\n{tb}"
            ) from e

        # Normalize to dict
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


# -------------------------------------------------------------------
# Preserved "LinkupJobSearch" architecture (for app.py + research intake)
# -------------------------------------------------------------------

class LinkupJobSearch:
    def __init__(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        api_key: str | None = None,
    ):
        _ = session_id
        _ = user_id
        self.client = LinkupClient(api_key=api_key)
        self.company_research_agent = CompanyResearchAgent(self.client)

    def build_job_intake(self, selected_jd_payload: dict) -> JobPostingIntake:
        """
        Normalize the "user selected one JD" payload into a stable intake object.

        Expected shape (example):
          { "answer": "...", "sources": [ { "url": "...", "snippet": "...", ... } ] }
        """
        return JobPostingIntake.from_selected_jd_payload(selected_jd_payload)

    def research_from_selected_jd(self, selected_jd_payload: dict) -> dict:
        """
        Convenience method: build the intake and run company research using it as context.
        Missing fields remain literal "NA" (no re-search / enrichment).
        """
        intake = self.build_job_intake(selected_jd_payload)

        company = intake.company_name if intake.company_name != "NA" else "NA"
        role = None if intake.role_title == "NA" else intake.role_title
        job_url = None if intake.job_url == "NA" else intake.job_url
        job_description = None if intake.answer == "NA" else intake.answer

        return self.company_research_agent.research_company(
            company=company,
            role=role,
            job_url=job_url,
            job_description=job_description,
            job_intake=intake,
        )

    def search_jobs(self, role: str, company: str | None = None, location: str = "United States") -> dict:
        today = datetime.now().strftime("%B %d, %Y")
        company_name = company or "top tech companies"

        query = f"""You are a job search specialist. Your objective is to find all current {role} job openings at {company_name} in {location} that are posted today or very recently.

- Today: {today}
- Only include roles posted today or within the last 7 days.
- Prioritize official {company_name} career pages when possible.

For each role found, extract:
- Job title
- Location
- Posting URL
- Posting date (verify recent)
- Key requirements / summary
- Salary range (if available)
"""

        return self.client.search(query=query, depth="deep", output_type="searchResults", max_results=12)

    def get_company_profile(self, company: str, query: str | None = None, *, context: Optional[Dict[str, Any]] = None) -> dict:
        query = query or f"{company} company overview funding tech stack culture engineering team 2025"
        _ = self.client.search(query=query, depth="deep", output_type="searchResults", max_results=8)
        return self.company_research_agent.research_profile(company, context=context)

    def get_company_sentiment(self, company: str, query: str | None = None, *, context: Optional[Dict[str, Any]] = None) -> dict:
        query = query or f"{company} employee reviews glassdoor engineering culture work life balance"
        _ = self.client.search(query=query, depth="standard", output_type="searchResults", max_results=8)
        return self.company_research_agent.research_sentiment(company, context=context)

    def find_recruiters(self, company: str, role: str, query: str | None = None) -> dict:
        query = query or f"{company} recruiter hiring manager {role} LinkedIn"
        return self.client.search(query=query, depth="standard", output_type="searchResults", max_results=8)
