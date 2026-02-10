from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
from pathlib import Path

from document_handler import DocumentHandler
from linkup_client import linkup_search
from document.pdf_export import export_resume_to_pdf_ats

# Optional DOCX export
try:
    from docx import Document as DocxDocument  # python-docx
except Exception:
    DocxDocument = None


@dataclass
class TailorRequest:
    resume_path: str
    job_description: str  # can be raw text OR a URL
    job_title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    output_docx_path: Optional[str] = None  # e.g., "outputs/tailored_resume.docx"
    output_pdf_path: Optional[str] = None   # e.g., "outputs/tailored_resume_ats.pdf"
    max_bullets_per_role: int = 3


URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _normalize(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _is_url(s: str) -> bool:
    return bool(URL_RE.match((s or "").strip()))


def _safe_text(s: Any) -> str:
    return str(s or "").strip()


# ---------------------------
# Job description acquisition
# ---------------------------
def _fetch_job_description_if_url(jd_or_url: str) -> str:
    """
    If the user provides a URL, use Linkup to open and extract the job description text.
    Otherwise treat as plain text.
    """
    jd_or_url = (jd_or_url or "").strip()
    if not _is_url(jd_or_url):
        return jd_or_url

    query = (
        "Open this job posting and extract the full job description text. "
        "Return only the job description content (responsibilities, requirements, skills). "
        f"URL: {jd_or_url}"
    )

    resp = linkup_search(query=query, depth="deep", output_type="searchResults", max_results=3)

    # Best-effort extraction of text fields from Linkup response
    if hasattr(resp, "model_dump"):
        resp = resp.model_dump()

    if isinstance(resp, dict):
        results = resp.get("results") or resp.get("data") or resp.get("output") or resp
        if isinstance(results, list) and results:
            r0 = results[0]
            for key in ("content", "text", "snippet", "summary"):
                if isinstance(r0, dict) and r0.get(key):
                    return str(r0[key])
        return str(resp)[:8000]

    return str(resp)[:8000]


# ---------------------------
# Skill/keyword extraction
# ---------------------------
TECH_TOKENS = {
    "python", "sql", "java", "javascript", "typescript", "c#", "c++", "go", "ruby",
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
    "spark", "hadoop", "airflow", "dbt",
    "pandas", "numpy",
    "postgres", "mysql", "mongodb", "snowflake",
    "tableau", "power bi", "excel",
    "rest", "restful", "api", "apis", "microservices",
    "react", "node", "flask", "django", ".net", "asp.net",
    "etl", "ml", "machine learning", "nlp",
}


def extract_jd_keywords(jd_text: str, top_k: int = 20) -> List[str]:
    """
    Lightweight, local-first keyword extraction:
    - matches known tech tokens
    - also grabs capitalized acronyms like "SQL", "AWS"
    """
    t = " " + _normalize(jd_text) + " "
    found = set()

    for tok in TECH_TOKENS:
        pattern = r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])"
        if re.search(pattern, t):
            found.add(tok)

    acronyms = re.findall(r"\b[A-Z]{2,7}(?:/[A-Z]{2,7})?\b", jd_text or "")
    for a in acronyms:
        al = a.lower()
        if al in {"us", "ny", "ca"}:
            continue
        if al in {"aws", "sql", "etl", "api", "apis", "ci/cd", "sdlc", "ml", "nlp"}:
            found.add(al)

    counts: List[Tuple[str, int]] = []
    for k in found:
        counts.append((k, len(re.findall(re.escape(k), t))))
    counts.sort(key=lambda x: x[1], reverse=True)

    return [k for k, _ in counts[:top_k]]


# ---------------------------
# Resume extraction helpers
# ---------------------------
def _get_resume_profile(resume_path: str) -> Dict[str, Any]:
    r = DocumentHandler().analyze(resume_path, doc_type="resume")
    if r.get("status") != "ok":
        raise RuntimeError(r.get("summary", "Resume analysis failed."))
    return r["data"]


def _resume_skill_list(resume_data: Dict[str, Any]) -> List[str]:
    skills = resume_data.get("skills_flat")
    if isinstance(skills, list) and skills:
        return [str(s) for s in skills]

    text = _safe_text(resume_data.get("text"))
    t = " " + _normalize(text) + " "
    found: List[str] = []
    for tok in sorted(TECH_TOKENS, key=len, reverse=True):
        if re.search(r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])", t):
            found.append(tok)

    seen = set()
    out: List[str] = []
    for s in found:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _score_bullet(bullet: str, jd_keys: List[str]) -> int:
    b = _normalize(bullet)
    score = 0
    for k in jd_keys:
        if k and k in b:
            score += 1
    return score


def _pick_top_bullets(
    entries: List[Dict[str, Any]],
    jd_keys: List[str],
    max_per_role: int,
) -> List[Dict[str, Any]]:
    """
    Keep each role, but select the top N bullets most aligned to the JD.
    """
    out: List[Dict[str, Any]] = []
    for e in entries or []:
        bullets = e.get("bullets") or []
        scored = sorted(
            [(b, _score_bullet(b, jd_keys)) for b in bullets],
            key=lambda x: x[1],
            reverse=True,
        )
        top = [b for b, _ in scored[:max_per_role] if b]
        item = dict(e)
        item["bullets"] = top
        out.append(item)
    return out


# ---------------------------
# Tailoring logic (no hallucinations)
# ---------------------------
def build_target_summary(
    contact_name: Optional[str],
    job_title: Optional[str],
    jd_keys: List[str],
    resume_keys: List[str],
) -> str:
    """
    Controlled summary generation:
    - Uses ONLY overlap between JD keywords and resume skills (no invented skills)
    - Clean fallback if overlap is empty
    """
    name = contact_name or "Candidate"
    title = job_title or "the role"

    resume_set = set([_normalize(x) for x in resume_keys if x])
    jd_norm = [_normalize(x) for x in jd_keys if x]

    overlap = [k for k in jd_norm if k in resume_set]
    top = overlap[:6]  # only overlap to avoid inventing skills

    skills_phrase = ", ".join([k.upper() if k in {"sql", "aws"} else k for k in top]) if top else ""

    if skills_phrase:
        return (
            f"{name} is targeting {title} and brings demonstrated experience with {skills_phrase}, "
            "along with a strong foundation in problem-solving, collaboration, and ownership."
        )

    # Fallback when there is NO real skill overlap
    return (
        f"{name} is targeting {title} and brings strong analytical thinking and communication skills, "
        "with hands-on experience delivering results through teamwork and structured problem-solving."
    )


def build_skills_section(jd_keys: List[str], resume_keys: List[str], max_items: int = 12) -> List[str]:
    rset = set([_normalize(x) for x in resume_keys if x])

    jd_norm = []
    for x in jd_keys:
        x2 = _normalize(x)
        if x2:
            jd_norm.append(x2)

    overlap = [k for k in jd_norm if k in rset]
    extras = []
    seen_overlap = set(overlap)
    for rk in resume_keys:
        rk2 = _normalize(rk)
        if rk2 and rk2 not in seen_overlap:
            extras.append(rk2)

    combined = (overlap + extras)[:max_items]

    seen = set()
    out: List[str] = []
    for s in combined:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def render_resume_text(resume_data: Dict[str, Any], tailored: Dict[str, Any]) -> str:
    contact = resume_data.get("contact") or {}
    name = contact.get("name") or ""
    email = contact.get("email") or ""
    phone = contact.get("phone") or ""
    location = contact.get("location") or ""
    links = contact.get("links") or {}

    header_line = " | ".join([x for x in [email, phone, location] if x])
    links_line = " | ".join([v for _, v in links.items()]) if isinstance(links, dict) else ""

    lines: List[str] = []
    if name:
        lines.append(name.upper())
    if header_line:
        lines.append(header_line)
    if links_line:
        lines.append(links_line)
    lines.append("")

    lines.append("SUMMARY")
    lines.append(tailored["summary"])
    lines.append("")

    lines.append("SKILLS")
    lines.append(", ".join(tailored["skills"]))
    lines.append("")

    lines.append("EXPERIENCE")
    for e in tailored["experience"]:
        title = e.get("title") or ""
        company = e.get("company") or ""
        start = e.get("start_date") or ""
        end = e.get("end_date") or ""
        head = " — ".join([x for x in [title, company] if x]).strip()
        date = " to ".join([x for x in [start, end] if x]).strip()

        if head and date:
            lines.append(f"{head} ({date})")
        elif head:
            lines.append(head)
        elif date:
            lines.append(date)

        for b in e.get("bullets") or []:
            lines.append(f"- {b}")
        lines.append("")

    sections = resume_data.get("sections") or {}
    edu = sections.get("education")
    if edu:
        lines.append("EDUCATION")
        lines.append(_safe_text(edu))
        lines.append("")

    return "\n".join(lines).strip()


def export_docx(text: str, output_path: str) -> str:
    if DocxDocument is None:
        raise RuntimeError("python-docx is not installed, cannot export DOCX.")
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    doc = DocxDocument()
    for para in text.split("\n"):
        doc.add_paragraph(para)
    doc.save(str(outp))
    return str(outp)


def tailor_resume(req: TailorRequest) -> Dict[str, Any]:
    """
    Main entry point for the Resume Tailor Agent.
    Generates tailored resume text + optional DOCX + optional ATS PDF.
    """
    resume_data = _get_resume_profile(req.resume_path)
    jd_text = _fetch_job_description_if_url(req.job_description)

    jd_keys = extract_jd_keywords(jd_text, top_k=20)
    resume_keys = _resume_skill_list(resume_data)

    summary = build_target_summary(
        contact_name=(resume_data.get("contact") or {}).get("name"),
        job_title=req.job_title,
        jd_keys=jd_keys,
        resume_keys=resume_keys,
    )

    skills = build_skills_section(jd_keys=jd_keys, resume_keys=resume_keys, max_items=12)

    experience_entries = resume_data.get("experience") or []
    volunteer_entries = resume_data.get("volunteer_experience") or []
    combined_entries = list(experience_entries) + list(volunteer_entries)

    tailored_entries = _pick_top_bullets(combined_entries, jd_keys, req.max_bullets_per_role)

    tailored = {
        "summary": summary,
        "skills": skills,
        "experience": tailored_entries,
    }

    resume_text = render_resume_text(resume_data, tailored)

    docx_path = None
    pdf_path = None

    if req.output_docx_path:
        docx_path = export_docx(resume_text, req.output_docx_path)

    if req.output_pdf_path:
        pdf_path = export_resume_to_pdf_ats(resume_text, req.output_pdf_path)

    return {
        "status": "ok",
        "type": "tailored_resume",
        "summary": "Tailored resume generated from user resume + job description.",
        "data": {
            "resume_path": req.resume_path,
            "job_title": req.job_title,
            "company": req.company,
            "location": req.location,
            "jd_keywords": jd_keys,
            "resume_skills": resume_keys,
            "tailored_resume_text": resume_text,
            "output_docx_path": docx_path,
            "output_pdf_path": pdf_path,
        },
        "confidence": {
            "tailoring": 0.7 if jd_keys else 0.4,
        },
        "citations": [],
    }
