from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

def extract_text(document_path: str) -> Dict[str, Any]:
    """
    Extract text from PDF/DOCX/TXT. Returns a small envelope with stats + preview.
    Local-first, no OCR yet.
    """
    path = Path(document_path)
    if not path.exists():
        return {"status": "error", "summary": f"File not found: {document_path}", "text": ""}

    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf_text(path)
    if ext == ".docx":
        return _extract_docx_text(path)
    if ext in {".txt", ".md"}:
        text = path.read_text(errors="ignore")
        return _wrap_ok(text, source_type=ext[1:])

    return {"status": "error", "summary": f"Unsupported file type: {ext}", "text": ""}


def _extract_pdf_text(path: Path) -> Dict[str, Any]:
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text
        text = pdf_extract_text(str(path)) or ""
        return _wrap_ok(text, source_type="pdf")
    except ModuleNotFoundError as e:
        # pdfminer.six provides the `pdfminer` module.
        if getattr(e, "name", "") == "pdfminer":
            return {
                "status": "error",
                "summary": (
                    "Missing dependency 'pdfminer.six' required for PDF text extraction.\n"
                    "\n"
                    "Install it with one of:\n"
                    "  - python -m pip install pdfminer.six\n"
                    "  - conda install -c conda-forge pdfminer.six\n"
                    "\n"
                    "If pip fails with an SSL certificate error on macOS/Anaconda (e.g. OSStatus -26276), "
                    "prefer the conda-forge install or update certs:\n"
                    "  - conda update -n base -c conda-forge ca-certificates certifi openssl\n"
                ),
                "text": "",
            }
        return {"status": "error", "summary": f"PDF text extraction failed: {e}", "text": ""}
    except Exception as e:
        return {"status": "error", "summary": f"PDF text extraction failed: {e}", "text": ""}


def _extract_docx_text(path: Path) -> Dict[str, Any]:
    try:
        import docx  # python-docx
        doc = docx.Document(str(path))
        parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n".join(parts)
        return _wrap_ok(text, source_type="docx")
    except ModuleNotFoundError as e:
        if getattr(e, "name", "") in {"docx", "python_docx"}:
            return {
                "status": "error",
                "summary": (
                    "Missing dependency 'python-docx' required for DOCX text extraction.\n"
                    "\n"
                    "Install it with:\n"
                    "  - python -m pip install python-docx\n"
                    "  - conda install -c conda-forge python-docx\n"
                ),
                "text": "",
            }
        return {"status": "error", "summary": f"DOCX text extraction failed: {e}", "text": ""}
    except Exception as e:
        return {"status": "error", "summary": f"DOCX text extraction failed: {e}", "text": ""}


def _wrap_ok(text: str, *, source_type: str) -> Dict[str, Any]:
    cleaned = text.strip()
    preview = cleaned[:800] + ("..." if len(cleaned) > 800 else "")
    return {
        "status": "ok",
        "type": "extracted_text",
        "source_type": source_type,
        "text_len": len(cleaned),
        "preview": preview,
        "text": cleaned,
    }
