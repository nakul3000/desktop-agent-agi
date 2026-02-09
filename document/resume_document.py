from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import re

from document.extract_text import extract_text


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

# Simple signals for link classification
LINK_LABELS = [
    ("linkedin", re.compile(r"linkedin\.com", re.IGNORECASE)),
    ("github", re.compile(r"github\.com", re.IGNORECASE)),
    ("portfolio", re.compile(r"portfolio|about|site|website|vercel|netlify", re.IGNORECASE)),
]


def _clean_line(line: str) -> str:
    return " ".join(line.strip().split())


def _extract_contact_block(text: str, max_lines: int = 12) -> str:
    """
    Resumes usually put contact info in the first few lines.
    We'll use the top N lines as a high-signal 'contact block'.
    """
    lines = [ln for ln in text.splitlines() if _clean_line(ln)]
    top = lines[:max_lines]
    return "\n".join(top)


def _extract_name_heuristic(contact_block: str) -> Optional[str]:
    """
    Heuristic: candidate name is often a standalone line near the top,
    in Title Case or ALL CAPS, not containing email/phone/url.
    """
    lines = [_clean_line(ln) for ln in contact_block.splitlines() if _clean_line(ln)]
    # Skip lines that clearly contain contact tokens
    filtered = []
    for ln in lines:
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln) or URL_RE.search(ln):
            continue
        # avoid headings like "EXAMPLE RESUME"
        if len(ln) > 45:
            continue
        filtered.append(ln)

    # Prefer the first plausible line with 2-4 words
    for ln in filtered[:6]:
        words = ln.replace(",", " ").split()
        if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
            return ln.title() if ln.isupper() else ln

    return None


def _extract_links(text: str) -> Dict[str, str]:
    links: Dict[str, str] = {}
    candidates = URL_RE.findall(text)

    # Normalize candidates
    norm = []
    for u in candidates:
        u = u.strip().rstrip(".,;")
        if u.lower().startswith("www."):
            u = "https://" + u
        norm.append(u)

    # Classify
    for u in norm:
        label = "other"
        for key, pattern in LINK_LABELS:
            if pattern.search(u):
                label = key
                break

        # keep first of each label
        if label not in links:
            links[label] = u

    # If "other" is the only one, keep it; otherwise drop "other"
    if "other" in links and len(links) > 1:
        links.pop("other", None)

    return links


def _extract_contact_fields(text: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Returns:
      contact: {name, email, phone, location, links}
      confidence: per-field confidence
    """
    contact_block = _extract_contact_block(text)
    email = (EMAIL_RE.findall(contact_block) or EMAIL_RE.findall(text) or [None])[0]
    phone = (PHONE_RE.findall(contact_block) or PHONE_RE.findall(text) or [None])[0]

    # Basic location heuristic: find a line with city/state pattern in contact block
    location = None
    lines = [_clean_line(ln) for ln in contact_block.splitlines() if _clean_line(ln)]
    # Example matches: "Modesto, CA" or "Modesto, CA, 71234"
    loc_re = re.compile(r"\b([A-Za-z .'-]+),\s*([A-Z]{2})(?:\s*,?\s*\d{5})?\b")
    for ln in lines:
        m = loc_re.search(ln)
        if m:
            location = f"{m.group(1).strip()}, {m.group(2).strip()}"
            break

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


def process_resume(document_path: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resume domain entry point.
    Step A: Extract raw text (local-first).
    Step B: Extract contact info (rules + heuristics).
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

    if text_len < 200:
        note = "Very little text extracted. Resume may be image-based; OCR fallback not enabled yet."
        text_conf = 0.4
    else:
        note = "Resume text extracted successfully."
        text_conf = 0.9

    confidence = {"text_extraction": text_conf, **contact_conf}

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
            # Keep full text for next steps (sectioning, experience extraction, etc.)
            "text": text,
        },
        "confidence": confidence,
        "citations": [],
    }
