from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

from pydantic import BaseModel, Field

from linkup_client import linkup_search


# -------------------------
# Models (structured output)
# -------------------------
class JobPosting(BaseModel):
    title: str
    company: str
    url: str

    location: Optional[str] = None
    remote: Optional[bool] = None
    date_posted: Optional[str] = None
    summary: Optional[str] = None
    requirements: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    source: Optional[str] = None


class JobSearchOutput(BaseModel):
    jobs: List[JobPosting] = Field(default_factory=list)


@dataclass
class JobSearchQuery:
    role: str
    company: Optional[str] = None
    location: str = "United States"
    remote_ok: bool = True
    posted_within_days: int = 30
    max_results: int = 15


# -------------------------
# Helpers
# -------------------------
URL_RE = re.compile(r"^https?://", re.IGNORECASE)

def _normalize(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _extract_jobs(resp: Any) -> List[Dict[str, Any]]:
    """
    Linkup may return structured results in a few shapes.
    Normalize them into a list[dict].
    """
    if hasattr(resp, "model_dump"):
        resp = resp.model_dump()

    if isinstance(resp, dict):
        # expected: {"jobs":[...]}
        if isinstance(resp.get("jobs"), list):
            return resp["jobs"]

        # sometimes nested: {"output":{"jobs":[...]}}
        out = resp.get("output")
        if isinstance(out, dict) and isinstance(out.get("jobs"), list):
            return out["jobs"]

        # sometimes nested: {"data":{"jobs":[...]}}
        data = resp.get("data")
        if isinstance(data, dict) and isinstance(data.get("jobs"), list):
            return data["jobs"]

    return []


def _dedupe_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for j in jobs:
        key = (
            _normalize(j.get("company")),
            _normalize(j.get("title")),
            _normalize(j.get("location")),
            _normalize(j.get("url")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(j)
    return out


# -------------------------
# Public API
# -------------------------
class LinkupJobSearch:
    """
    A focused job search utility built on your stable `linkup_search()` wrapper.

    Use cases:
    - Find job postings
    - Fetch job description text from a selected job URL
    """

    def search_jobs(self, q: JobSearchQuery) -> Dict[str, Any]:
        company_part = f'at "{q.company}" ' if q.company else ""
        remote_part = " remote" if q.remote_ok else ""
        loc_part = f"in {q.location}" if q.location else ""

        # IMPORTANT: recency is enforced in prompt text (since SDK versions vary)
        query = (
            f"Find currently open job postings for \"{q.role}\" {company_part}{remote_part} {loc_part}. "
            f"Only include roles posted in the last {q.posted_within_days} days if possible. "
            f"Prefer official company career pages over aggregators (LinkedIn/Indeed OK only if needed). "
            f"Open each job posting page and extract structured job fields. "
            f"Return direct job posting URLs (not search result pages)."
        )

        resp = linkup_search(
            query=query,
            depth="deep",
            output_type="structured",
            schema=JobSearchOutput,
            max_results=q.max_results,
            recency_days=q.posted_within_days,  # client keeps API stable; prompt enforces if ignored
        )

        jobs = _extract_jobs(resp)
        jobs = _dedupe_jobs(jobs)

        return {
            "status": "ok",
            "type": "linkup_job_search_results",
            "summary": f"Found {len(jobs)} job postings.",
            "data": {
                "query": {
                    "role": q.role,
                    "company": q.company,
                    "location": q.location,
                    "remote_ok": q.remote_ok,
                    "posted_within_days": q.posted_within_days,
                    "max_results": q.max_results,
                },
                "jobs": jobs,
                "raw": resp,
            },
            "confidence": {"retrieval": 0.75 if jobs else 0.3},
            "citations": [],
        }

    def fetch_job_description(self, job_url_or_text: str, *, max_chars: int = 12000) -> Dict[str, Any]:
        """
        If given a URL: opens it via Linkup and extracts JD text.
        If given raw text: returns it as-is.

        This is useful for your flow:
        user selects recommended job -> fetch JD -> tailor resume -> export ATS PDF
        """
        s = (job_url_or_text or "").strip()
        if not s:
            return {
                "status": "error",
                "type": "job_description",
                "summary": "Empty job description / URL provided.",
                "data": {"job_description": ""},
                "confidence": {"extraction": 0.1},
                "citations": [],
            }

        if not URL_RE.match(s):
            return {
                "status": "ok",
                "type": "job_description",
                "summary": "Job description provided as text.",
                "data": {"job_description": s[:max_chars], "source": "user_text"},
                "confidence": {"extraction": 0.9},
                "citations": [],
            }

        query = (
            "Open this job posting and extract the full job description text. "
            "Return ONLY the job description content (responsibilities, requirements, qualifications, skills). "
            f"URL: {s}"
        )

        resp = linkup_search(query=query, depth="deep", output_type="searchResults", max_results=3)

        if hasattr(resp, "model_dump"):
            resp = resp.model_dump()

        jd_text = ""
        if isinstance(resp, dict):
            results = resp.get("results") or resp.get("data") or resp.get("output") or []
            if isinstance(results, list) and results:
                r0 = results[0]
                if isinstance(r0, dict):
                    for key in ("content", "text", "snippet", "summary"):
                        if r0.get(key):
                            jd_text = str(r0[key])
                            break

        if not jd_text:
            # last resort
            jd_text = str(resp)

        jd_text = jd_text[:max_chars]

        return {
            "status": "ok",
            "type": "job_description",
            "summary": "Extracted job description from URL.",
            "data": {"job_description": jd_text, "source": s, "raw": resp},
            "confidence": {"extraction": 0.65 if jd_text else 0.2},
            "citations": [],
        }
