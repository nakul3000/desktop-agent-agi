from __future__ import annotations
from typing import Any, Dict, Optional
from pathlib import Path

from document.resume_document import process_resume
from document.extract_text import extract_text


class DocumentHandler:
    def analyze(
        self,
        document_path: str,
        *,
        doc_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a document and return a standardized result envelope.

        doc_type:
          - "resume" (explicit)
          - None → inferred by filename for now (simple heuristic)

        For non-resume docs, falls back to generic local text extraction.
        """
        context = context or {}
        path = Path(document_path)

        if not path.exists():
            return {
                "status": "error",
                "type": "file",
                "summary": f"File not found: {document_path}",
                "data": {},
                "confidence": {},
                "citations": [],
            }

        inferred = (doc_type or "").strip().lower() or self._infer_doc_type(path.name)

        if inferred == "resume":
            return process_resume(str(path), context=context)

        # Fallback: generic text extraction for PDFs/DOCX/TXT
        extracted = extract_text(str(path))
        if extracted.get("status") == "ok":
            return {
                "status": "ok",
                "type": "document_text",
                "summary": f"Extracted text from {path.name} ({extracted.get('source_type')}).",
                "data": {
                    "file_name": path.name,
                    "file_path": str(path),
                    "text_len": extracted.get("text_len"),
                    "preview": extracted.get("preview"),
                    # Keep full text available for downstream use
                    "text": extracted.get("text", ""),
                },
                "confidence": {},
                "citations": [],
            }

        return {
            "status": "error",
            "type": inferred or "unknown",
            "summary": extracted.get("summary", f"Unsupported document type: {inferred}"),
            "data": {"file_path": str(path)},
            "confidence": {},
            "citations": [],
        }

    @staticmethod
    def _infer_doc_type(filename: str) -> str:
        """
        Very light heuristic for now.
        We’ll improve later using content-based classification.
        """
        name = filename.lower()
        if "resume" in name or "cv" in name:
            return "resume"
        return "unknown"
