from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path
import re

from document.extract_text import extract_text

# -------------------------
# Contact extraction regexes
# -------------------------
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(
    r"""
    (?:
        (?:\+?1[\s.-]?)?              # optional country code
        (?:\(?\d{3}\)?[\s.-]?)        # area code
        \d{3}[\s.-]?\d{4}             # local number
    )
    """,
    re.VERBOSE,
)
URL_RE = re.compile(r"\bhttps?://[^\s)]+|\bwww\.[^\s)]+", re.IGNORECASE)

LINK_LABELS = [
    ("linkedin", re.compile(r"linkedin\.com", re.IGNORECASE)),
    ("github", re.compile(r"github\.com", re.IGNORECASE)),
    ("portfolio", re.compile(r"portfolio|about|site|website|vercel|netlify", re.IGNORECASE)),
]


def _clean_line(line: str) -> str:
    return " ".join(line.strip().split())


def _extract_contact_block(text: str, max_lines: int = 12) -> str:
    lines = [ln for ln in text.splitlines() if _clean_line(ln)]
    return "\n".join(lines[:max_lines])


def _extract_name_heuristic(contact_block: str) -> Optional[str]:
    bad_tokens = {"resume", "curriculum", "vitae", "cv"}
    lines = [_clean_line(ln) for ln in contact_block.splitlines() if _clean_line(ln)]
    candidates: List[str] = []

    for ln in lines:
        low = ln.lower()
        if any(tok in low for tok in bad_tokens):
            continue
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln) or URL_RE.search(ln):
            continue
        if len(ln) > 45:
            continue
        candidates.append(ln)

    for ln in candidates[:8]:
        words = ln.replace(",", " ").split()
        if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
            return ln.title() if ln.isupper() else ln

    return None


def _extract_links(text: str) -> Dict[str, str]:
    links: Dict[str, str] = {}
    candidates = URL_RE.findall(text)

    norm: List[str] = []
    for u in candidates:
        u = u.strip().rstrip(".,;")
        if u.lower().startswith("www."):
            u = "https://" + u
        norm.append(u)

    for u in norm:
        label = "other"
        for key, pattern in LINK_LABELS:
            if pattern.search(u):
                label = key
                break
        if label not in links:
            links[label] = u

    if "other" in links and len(links) > 1:
        links.pop("other", None)

    return links


def _extract_contact_fields(text: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
    contact_block = _extract_contact_block(text)

    email = (EMAIL_RE.findall(contact_block) or EMAIL_RE.findall(text) or [None])[0]
    phone = (PHONE_RE.findall(contact_block) or PHONE_RE.findall(text) or [None])[0]

    location = None
    lines = [_clean_line(ln) for ln in contact_block.splitlines() if _clean_line(ln)]
    loc_re = re.compile(r"\b([A-Za-z][A-Za-z .'-]+?),\s*([A-Z]{2})\b")

    matches = []
    for ln in lines:
        for m in loc_re.finditer(ln):
            matches.append((ln, m.group(1).strip(), m.group(2).strip()))

    if matches:
        _, city_raw, state = matches[-1]
        city = city_raw.split(".")[-1].strip()
        location = f"{city}, {state}"

    name = _extract_name_heuristic(contact_block)
    links = _extract_links(text)

    confidence = {
        "contact.name": 0.75 if name else 0.2,
        "contact.email": 0.95 if email else 0.1,
        "contact.phone": 0.9 if phone else 0.1,
        "contact.location": 0.7 if location else 0.2,
        "contact.links": 0.85 if links else 0.2,
    }

    contact = {
        "name": name,
        "email": email,
        "phone": phone,
        "location": location,
        "links": links,
    }
    return contact, confidence


# -------------------------
# Sectioning
# -------------------------
SECTION_ALIASES = {
    "objective": ["objective", "summary", "professional summary"],
    "education": ["education", "academics"],
    "experience": ["work experience", "experience", "employment", "work"],
    "volunteer_experience": ["volunteer experience", "volunteering"],
    "projects": ["projects", "project experience"],
    "skills": ["skills", "technical skills", "core skills"],
    "activities": ["activities", "extracurricular", "achievements", "honors", "honors and activities"],
}


def _normalize_heading(s: str) -> str:
    s = _clean_line(s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def _is_heading_line(line: str) -> bool:
    ln = _clean_line(line)
    if not ln:
        return False
    if EMAIL_RE.search(ln) or PHONE_RE.search(ln) or URL_RE.search(ln):
        return False
    if len(ln) > 45:
        return False

    letters = [c for c in ln if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    ratio = upper / max(1, len(letters))
    return ratio >= 0.75


def _map_heading_to_section(heading: str) -> Optional[str]:
    h = _normalize_heading(heading)
    for section, aliases in SECTION_ALIASES.items():
        for a in aliases:
            if a in h:
                return section
    return None


def split_resume_sections(text: str) -> Dict[str, str]:
    lines = list(text.splitlines())
    current_section = "header"
    buckets: Dict[str, List[str]] = {"header": []}

    for raw in lines:
        ln = _clean_line(raw)
        if not ln:
            continue

        if _is_heading_line(ln):
            mapped = _map_heading_to_section(ln)
            if mapped:
                current_section = mapped
                buckets.setdefault(current_section, [])
                continue

        buckets.setdefault(current_section, [])
        buckets[current_section].append(ln)

    out: Dict[str, str] = {}
    for k, v in buckets.items():
        block = "\n".join(v).strip()
        if block:
            out[k] = block
    return out


# -------------------------
# Experience parsing (OCR-tolerant)
# -------------------------
MONTHS = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

PRESENT_RE = re.compile(r"\b(present|current)\b", re.IGNORECASE)


def _normalize_month(m: str) -> Optional[str]:
    key = re.sub(r"[^a-z]", "", (m or "").lower())
    return MONTHS.get(key)


def _clean_year(y: str) -> Optional[str]:
    y = (y or "").strip()
    y = y.replace("&", "8").replace("O", "0").replace("I", "1").replace("l", "1")
    y = re.sub(r"[^\d]", "", y)
    if len(y) == 4 and y.isdigit():
        return y
    return None


DATE_RANGE_RE = re.compile(
    r"""
    (?P<start_month>[A-Za-z]{3,9})\s+(?P<start_year>[0-9OIl&]{4})
    \s*(?:-|to|TO|–|—)\s*
    (?:
        (?P<end_month>[A-Za-z]{3,9})\s+(?P<end_year>[0-9OIl&]{4})
        |PRESENT|Present|present
    )
    """,
    re.VERBOSE,
)


def _parse_date_range(line: str) -> Tuple[Optional[str], Optional[str]]:
    m = DATE_RANGE_RE.search(line)
    if not m:
        return None, None

    sm = _normalize_month(m.group("start_month"))
    sy = _clean_year(m.group("start_year"))
    start = f"{sy}-{sm}" if (sy and sm) else None

    if PRESENT_RE.search(line):
        return start, "present"

    em_raw = m.group("end_month")
    ey_raw = m.group("end_year")
    em = _normalize_month(em_raw) if em_raw else None
    ey = _clean_year(ey_raw) if ey_raw else None
    end = f"{ey}-{em}" if (ey and em) else None

    return start, end


def _looks_like_experience_header(line: str) -> bool:
    ln = _clean_line(line)
    if not ln:
        return False
    start, end = _parse_date_range(ln)
    if not start and not end:
        return False
    left = re.split(r"[•|·]", ln)[0].strip()
    return len(left) >= 3


def parse_experience_section(section_text: str) -> List[Dict[str, Any]]:
    lines = [_clean_line(ln) for ln in section_text.splitlines() if _clean_line(ln)]
    entries: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for ln in lines:
        if ln in {"•", "-", "—", "–", "'"}:
            continue

        if _looks_like_experience_header(ln):
            if current:
                current["bullets"] = [b for b in current["bullets"] if b]
                entries.append(current)

            start, end = _parse_date_range(ln)

            title_part = re.split(r"[•|·]", ln)[0].strip()
            title_part = title_part.replace("  ", " ").strip(" -•·")

            current = {
                "title": title_part.title() if title_part.isupper() else title_part,
                "company": None,
                "start_date": start,
                "end_date": end,
                "bullets": [],
            }
            continue

        if current:
            cleaned = ln.lstrip("•").strip()
            if _is_heading_line(cleaned) and _map_heading_to_section(cleaned):
                continue
            current["bullets"].append(cleaned)

    if current:
        current["bullets"] = [b for b in current["bullets"] if b]
        entries.append(current)

    return entries


# -------------------------
# Skill inference (fallback when no SKILLS section)
# -------------------------
SKILL_KEYWORDS = {
    "technical": [
        "python", "java", "javascript", "typescript", "sql", "react", "node", "aws", "gcp", "azure",
        "docker", "kubernetes", "git", "linux", "spark", "pandas", "numpy", "mongodb", "postgres",
        "mysql", "snowflake", "tableau", "power bi", "excel",
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "collaboration", "time management",
        "problem solving", "customer service", "organization",
    ],
    "domain": [
        "data mining", "nlp", "machine learning", "analytics", "talent acquisition",
        "operations", "recruiting",
    ],
}

def infer_skills_from_text(text: str) -> Dict[str, List[str]]:
    """
    Very lightweight keyword scan to infer skills.
    Returns categorized skills, deduped.
    """
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    found: Dict[str, List[str]] = {k: [] for k in SKILL_KEYWORDS.keys()}

    for cat, skills in SKILL_KEYWORDS.items():
        for s in skills:
            # word-boundary-ish match
            pattern = r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])"
            if re.search(pattern, t):
                found[cat].append(s)

    # Dedup + stable sort
    for k in list(found.keys()):
        found[k] = sorted(set(found[k]))
        if not found[k]:
            found.pop(k, None)

    return found

def flatten_skills(skills_by_cat: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for _, items in skills_by_cat.items():
        out.extend(items)
    # unique preserving order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# -------------------------
# Main entry point
# -------------------------
def process_resume(document_path: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resume domain entry point.
    A: Extract raw text
    B: Contact info
    C: Sections + Experience parsing + Volunteer split
    D: Skills inference (fallback)
    """
    context = context or {}
    path = Path(document_path)

    if not path.exists():
        return {
            "status": "error",
            "type": "resume_profile",
            "summary": f"File not found: {document_path}",
            "data": {},
            "confidence": {},
            "citations": [],
        }

    extracted = extract_text(str(path))
    if extracted.get("status") != "ok":
        return {
            "status": "error",
            "type": "resume_profile",
            "summary": extracted.get("summary", "Resume text extraction failed."),
            "data": {"file_name": path.name, "file_path": str(path)},
            "confidence": {},
            "citations": [],
        }

    text = extracted.get("text", "") or ""
    text_len = extracted.get("text_len", len(text))
    preview = extracted.get("preview", "")

    contact, contact_conf = _extract_contact_fields(text)
    sections = split_resume_sections(text)

    experience_entries = parse_experience_section(sections.get("experience", ""))
    volunteer_entries = [e for e in experience_entries if "volunteer" in (e.get("title", "").lower())]
    experience_entries = [e for e in experience_entries if e not in volunteer_entries]

    # Skill inference from all major text blocks
    combined_for_skills = "\n".join(
        [
            sections.get("education", ""),
            sections.get("activities", ""),
            sections.get("experience", ""),
            sections.get("objective", ""),
        ]
    )
    skills_inferred = infer_skills_from_text(combined_for_skills)
    skills_flat = flatten_skills(skills_inferred)

    if text_len < 200:
        note = "Very little text extracted. Resume may be image-based; OCR fallback not enabled yet."
        text_conf = 0.4
    else:
        note = "Resume text extracted successfully."
        text_conf = 0.9

    confidence = {"text_extraction": text_conf, **contact_conf}
    if skills_flat:
        confidence["skills.inferred"] = 0.6
    else:
        confidence["skills.inferred"] = 0.2

    return {
        "status": "ok",
        "type": "resume_profile",
        "summary": note,
        "data": {
            "file_name": path.name,
            "file_path": str(path),
            "source_type": extracted.get("source_type"),
            "text_len": text_len,
            "preview": preview,
            "contact": contact,
            "sections": sections,
            "experience": experience_entries,
            "volunteer_experience": volunteer_entries,
            "skills_inferred": skills_inferred,
            "skills_flat": skills_flat,
            "text": text,
        },
        "confidence": confidence,
        "citations": [],
    }
