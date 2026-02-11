from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

from linkup_client import linkup_search


@dataclass
class JobDescriptionRequest:
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    max_variants: int = 3


def _normalize(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())

def _clean_jd_text(text: str) -> str:
    t = _normalize(text)

    # Drop common nav/boilerplate fragments
    junk_patterns = [
        r"\[skip to content\]\(#content\)",
        r"skip to content",
        r"cookie",
        r"equal opportunity",
        r"accommodation",
        r"privacy policy",
        r"terms of use",
    ]
    for p in junk_patterns:
        t = re.sub(p, " ", t, flags=re.IGNORECASE)

    # Collapse repeated headings noise
    t = re.sub(r"(#\s*)+", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def fetch_sample_job_descriptions(req: JobDescriptionRequest) -> Dict[str, Any]:
    """
    Job Description Agent:
    Given a selected job, fetch clean, sample job descriptions
    the user can choose from before resume tailoring.
    """

    # Build a very explicit agentic query
    query = (
        f"Find the full job description for the role '{req.title}'. "
    )

    if req.company:
        query += f"The company is {req.company}. "
    if req.location:
        query += f"The location is {req.location}. "
    if req.url:
        query += (
            f"Use this job posting URL as the primary source: {req.url}. "
        )

    query += (
        "Extract the complete job description including:\n"
        "- Responsibilities\n"
        "- Requirements\n"
        "- Skills\n\n"
        "If multiple similar versions exist (e.g., same role on company site and aggregator), "
        "return up to 3 clean job description variants. "
        "Return concise but complete descriptions suitable for resume tailoring."
    )

    # We don’t need structured schema here — text quality matters more
    resp = linkup_search(
        query=query,
        depth="deep",
        output_type="searchResults",
        max_results=req.max_variants,
    )

    # Normalize Linkup response
    if hasattr(resp, "model_dump"):
        resp = resp.model_dump()

    descriptions: List[Dict[str, Any]] = []

    results = resp.get("results") or resp.get("data") or resp.get("output") or []

    if isinstance(results, dict):
        results = [results]

    for r in results:
        if not isinstance(r, dict):
            continue

        text = (
            r.get("content")
            or r.get("text")
            or r.get("snippet")
            or r.get("summary")
        )

        if not text:
            continue

        descriptions.append({
            "source": r.get("source") or r.get("url"),
            "job_description": _clean_jd_text(text),
        })

        if len(descriptions) >= req.max_variants:
            break

    return {
        "status": "ok",
        "type": "job_description_samples",
        "summary": f"Retrieved {len(descriptions)} job description samples.",
        "data": {
            "title": req.title,
            "company": req.company,
            "location": req.location,
            "samples": descriptions,
        },
        "confidence": {
            "retrieval": 0.8 if descriptions else 0.3
        },
        "citations": [],
    }
