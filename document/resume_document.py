from __future__ import annotations
from typing import Any, Dict, Optional
from pathlib import Path

from document.extract_text import extract_text


def process_resume(document_path: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resume domain entry point.
    Step A: Extract raw text (local-first) and return it in a handler-friendly envelope.
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
            "data": {
                "file_name": path.name,
                "file_path": str(path),
            },
            "confidence": {},
            "citations": [],
        }

    text = extracted.get("text", "") or ""
    text_len = extracted.get("text_len", len(text))
    preview = extracted.get("preview", "")

    # If the PDF is image-based slides/scans, text_len may be tiny.
    # We’ll add OCR fallback in a later step.
    if text_len < 200:
        note = "Very little text extracted. Resume may be image-based; OCR fallback not enabled yet."
    else:
        note = "Resume text extracted successfully."

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
            # Keep full text for next steps (contact extraction, sectioning, LLM structuring)
            "text": text,
        },
        "confidence": {
            "text_extraction": 0.9 if text_len >= 200 else 0.4
        },
        "citations": [],
    }
