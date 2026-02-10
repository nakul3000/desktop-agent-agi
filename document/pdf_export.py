from __future__ import annotations

from pathlib import Path
from typing import List

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def export_resume_to_pdf_ats(
    resume_text: str,
    output_path: str,
    *,
    font_name: str = "Helvetica",
    font_size: int = 10,
    title_size: int = 14,
    line_gap: int = 3,
) -> str:
    """
    ATS-friendly PDF generator:
    - Single column
    - Text-based (not images)
    - No tables / no columns / no text boxes
    - Standard fonts
    - Simple bullets
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out), pagesize=LETTER)
    width, height = LETTER

    left = 0.75 * inch
    right = 0.75 * inch
    top = 0.75 * inch
    bottom = 0.75 * inch

    max_width = width - left - right

    def wrap_line(text: str, size: int) -> List[str]:
        """Hard-wrap a line to fit max_width using the current font."""
        if not text:
            return [""]
        words = text.split()
        lines: List[str] = []
        current = ""
        for w in words:
            trial = (current + " " + w).strip()
            if pdfmetrics.stringWidth(trial, font_name, size) <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines or [""]

    y = height - top

    def new_page():
        nonlocal y
        c.showPage()
        y = height - top

    def draw_text_line(text: str, size: int, bold: bool = False):
        nonlocal y
        if y <= bottom:
            new_page()

        # ATS-safe: Helvetica / Helvetica-Bold
        face = f"{font_name}-Bold" if bold and font_name == "Helvetica" else font_name
        c.setFont(face, size)
        c.drawString(left, y, text)
        y -= (size + line_gap)

    # Parse and write
    lines = [ln.rstrip() for ln in resume_text.split("\n")]

    name_written = False

    for raw in lines:
        ln = raw.strip()

        # Blank line
        if not ln:
            y -= (font_size + 2)
            continue

        # Name line: first non-empty line -> bigger
        if not name_written:
            for w in wrap_line(ln, title_size):
                draw_text_line(w, title_size, bold=True)
            name_written = True
            continue

        # Section heading: ALL CAPS and short
        if ln.isupper() and len(ln) <= 40:
            draw_text_line(ln, font_size + 1, bold=True)
            continue

        # Bullet handling (ATS-safe): "- " prefix
        if ln.startswith("- "):
            bullet_text = ln[2:].strip()
            wrapped = wrap_line(bullet_text, font_size)

            if wrapped:
                draw_text_line(f"- {wrapped[0]}", font_size, bold=False)
                for cont in wrapped[1:]:
                    if y <= bottom:
                        new_page()
                    c.setFont(font_name, font_size)
                    c.drawString(left + 14, y, cont)
                    y -= (font_size + line_gap)
            continue

        # Regular line wrap
        wrapped = wrap_line(ln, font_size)
        for w in wrapped:
            draw_text_line(w, font_size, bold=False)

    c.save()
    return str(out)