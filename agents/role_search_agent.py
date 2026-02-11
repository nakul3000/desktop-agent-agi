from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

from linkup_client import linkup_search
from document_handler import DocumentHandler

from pydantic import BaseModel, Field


# -------------------------
# Role Search (Linkup)
# -------------------------
@dataclass
class RoleSearchQuery:
    titles: List[str]
    location: Optional[str] = None
    remote_ok: bool = True
    keywords: Optional[List[str]] = None
    posted_within_days: int = 30
    max_results: int = 20


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


class RoleSearchOutput(BaseModel):
    jobs: List[JobPosting] = Field(default_factory=list)


def _normalize(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _dedupe_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
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


def _extract_jobs_from_linkup_response(resp: Any) -> List[Dict[str, Any]]:
    """
    Linkup SDK sometimes nests structured output under different keys.
    This tries common shapes.
    """
    if hasattr(resp, "model_dump"):
        resp = resp.model_dump()

    if isinstance(resp, dict):
        if isinstance(resp.get("jobs"), list):
            return resp["jobs"]

        out = resp.get("output")
        if isinstance(out, dict) and isinstance(out.get("jobs"), list):
            return out["jobs"]

        data = resp.get("data")
        if isinstance(data, dict) and isinstance(data.get("jobs"), list):
            return data["jobs"]

    return []


def search_roles(q: RoleSearchQuery) -> Dict[str, Any]:
    titles_part = " OR ".join([f"\"{t}\"" for t in q.titles])

    kw_part = ""
    if q.keywords:
        kw_part = " AND (" + " OR ".join([f"\"{k}\"" for k in q.keywords]) + ")"

    remote_part = " remote" if q.remote_ok else ""
    location_clause = f"in {q.location}" if q.location else ""

    query = (
        f"Find currently open job postings for {titles_part}{kw_part}{remote_part} {location_clause}. "
        f"Only include roles posted in the last {q.posted_within_days} days if possible. "
        f"Prefer official company career pages over aggregators (LinkedIn/Indeed OK only if needed). "
        f"Open each job posting page and extract structured job fields. "
        f"Return direct job posting URLs (not search result pages)."
    )

    resp = linkup_search(
        query=query,
        depth="deep",
        output_type="structured",
        schema=RoleSearchOutput,
        max_results=q.max_results,
        recency_days=q.posted_within_days,
    )

    jobs = _extract_jobs_from_linkup_response(resp)
    jobs = _dedupe_jobs(jobs)

    return {
        "status": "ok",
        "type": "role_search_results",
        "summary": f"Found {len(jobs)} job postings.",
        "data": {"query": q.__dict__, "jobs": jobs},
        "confidence": {"retrieval": 0.75 if jobs else 0.3},
        "citations": [],
    }


# -------------------------
# Resume-based recommendation
# -------------------------
DEFAULT_SKILL_KEYWORDS = {
    "python", "sql", "aws", "gcp", "azure", "docker", "kubernetes", "react", "javascript",
    "typescript", "java", "c#", "c++", "etl", "airflow", "spark", "pandas", "numpy",
    "tableau", "power bi", "excel", "snowflake", "postgres", "mysql", "mongodb",
}


def _extract_resume_skills(resume_path: str) -> List[str]:
    """
    Uses your local resume pipeline to get skills.
    Falls back to scanning resume text for DEFAULT_SKILL_KEYWORDS.
    Returns only technical skills useful for matching/search.
    """
    r = DocumentHandler().analyze(resume_path, doc_type="resume")
    if r.get("status") != "ok":
        return []

    data = r.get("data", {}) or {}

    skills = data.get("skills_flat") or []
    if not skills:
        text = (data.get("text") or "").lower()
        skills = [s for s in DEFAULT_SKILL_KEYWORDS if s in text]

    # Normalize and keep only technical keywords
    norm: List[str] = []
    for s in skills:
        s2 = _normalize(s)
        if not s2 or len(s2) < 2:
            continue
        norm.append(s2)

    filtered = [s for s in norm if s in DEFAULT_SKILL_KEYWORDS]

    # Dedupe preserve order
    seen = set()
    out: List[str] = []
    for s in filtered:
        if s not in seen:
            seen.add(s)
            out.append(s)

    return out


def _score_job(job: Dict[str, Any], resume_skills: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Score using overlap between resume_skills and job['skills'] + requirement text.
    Returns: (score_0_to_1, matched_skills, missing_skills)
    """
    rs = [_normalize(s) for s in resume_skills if s]
    rs_set = set(rs)

    job_skills = job.get("skills") or []
    js = [_normalize(s) for s in job_skills if s]
    js_set = set(js)

    req_text = _normalize(" ".join(job.get("requirements") or []))

    matched: List[str] = []
    for s in rs:
        if s in js_set or (s and s in req_text):
            matched.append(s)

    missing = [s for s in js if s and s not in rs_set]

    if js_set:
        overlap = len(set(matched) & js_set) / max(1, len(js_set))
    else:
        overlap = len(matched) / max(1, min(10, len(rs_set)))

    title = _normalize(job.get("title"))
    bonus = 0.05 if any(s in title for s in rs_set) else 0.0

    score = max(0.0, min(1.0, overlap + bonus))

    # unique preserve order
    seen = set()
    matched_u: List[str] = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            matched_u.append(m)

    seen2 = set()
    missing_u: List[str] = []
    for m in missing:
        if m not in seen2:
            seen2.add(m)
            missing_u.append(m)

    return score, matched_u, missing_u


def recommend_jobs_for_resume(
    resume_path: str,
    *,
    titles: Optional[List[str]] = None,
    location: Optional[str] = "United States",
    remote_ok: bool = True,
    posted_within_days: int = 60,
    max_results: int = 40,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    End-to-end:
    resume -> skills -> role search -> ranking -> threshold -> top_k
    """
    titles = titles or ["Software Engineer", "Data Analyst", "Data Engineer"]

    resume_skills = _extract_resume_skills(resume_path)
    keywords = resume_skills[:8] if resume_skills else None

    results = search_roles(
        RoleSearchQuery(
            titles=titles,
            location=location,
            remote_ok=remote_ok,
            keywords=keywords,
            posted_within_days=posted_within_days,
            max_results=max_results,
        )
    )

    jobs = results.get("data", {}).get("jobs", []) or []
    if not jobs:
        return {
            "status": "ok",
            "type": "job_recommendations",
            "summary": "No job postings retrieved from Linkup.",
            "data": {
                "resume_path": resume_path,
                "resume_skills_used": keywords or [],
                "recommendation_mode": "no_results",
                "threshold": 0.15,
                "recommendations": [],
            },
            "confidence": {"ranking": 0.2},
            "citations": [],
        }

    scored: List[Dict[str, Any]] = []
    for j in jobs:
        score, matched, missing = _score_job(j, resume_skills)
        item = dict(j)
        item["match_score"] = round(score, 3)
        item["matched_skills"] = matched
        item["missing_skills"] = missing[:10]
        scored.append(item)

    # Primary sort by match score
    scored.sort(key=lambda x: x.get("match_score", 0), reverse=True)

    THRESHOLD = 0.15
    filtered = [r for r in scored if r.get("match_score", 0) >= THRESHOLD]

    # Fallback ordering: title affinity first, then match score
    titles_norm = [_normalize(t) for t in titles]

    def title_affinity(job: Dict[str, Any]) -> int:
        jt = _normalize(job.get("title"))
        if any(t in jt for t in titles_norm):
            return 2
        if any(tok in jt for t in titles_norm for tok in t.split()):
            return 1
        return 0

    if filtered:
        chosen = filtered[:top_k]
        mode = "resume_matched"
        summary = f"Recommended {len(chosen)} jobs based on resume skill match."
        conf = 0.75
    else:
        scored.sort(key=lambda x: (title_affinity(x), x.get("match_score", 0)), reverse=True)
        chosen = scored[:top_k]
        mode = "generic_fallback"
        summary = (
            "No strong resume-skill matches found (match_score < 0.15). "
            f"Showing {len(chosen)} generic recommendations based on your target titles/location."
        )
        conf = 0.4

    return {
        "status": "ok",
        "type": "job_recommendations",
        "summary": summary,
        "data": {
            "resume_path": resume_path,
            "resume_skills_used": keywords or [],
            "recommendation_mode": mode,
            "threshold": THRESHOLD,
            "recommendations": [
                {
                    "title": j.get("title"),
                    "company": j.get("company"),
                    "location": j.get("location"),
                    "url": j.get("url"),
                    "match_score": j.get("match_score"),
                    "matched_skills": j.get("matched_skills"),
                    "missing_skills": j.get("missing_skills"),
                    "source": j.get("source"),
                    "date_posted": j.get("date_posted"),
                }
                for j in chosen
            ],
        },
        "confidence": {"ranking": conf},
        "citations": [],
    }
