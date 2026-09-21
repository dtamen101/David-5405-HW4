"""Turn slide decks and readings into note-sized chunks.

Supported formats: .pptx (one chunk per slide, speaker notes included),
.pdf (one chunk per page), .md / .txt (one chunk per heading section).

The python-pptx iteration pattern (``slide.shapes`` -> ``shape.has_text_frame``
-> ``slide.notes_slide.notes_text_frame``) is adapted from the python-pptx
docs "Getting Started" guide: https://python-pptx.readthedocs.io/en/latest/user/quickstart.html
The pypdf ``PdfReader(...).pages[i].extract_text()`` usage follows
https://pypdf.readthedocs.io/en/stable/user/extract-text.html
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .security import DocumentParseError, InvalidInputError, NotFoundError

SUPPORTED_SUFFIXES = {".pptx", ".pdf", ".md", ".txt"}
MAX_CHUNK_CHARS = 2000


@dataclass(frozen=True)
class Chunk:
    title: str
    text: str
    location: str  # e.g. "slide 4", "page 2", "section 3"


def resolve_source(sources_dir: Path, filename: str, max_mb: int) -> Path:
    """Map a client-supplied filename to a file *inside* ``sources_dir``.

    The client never learns the absolute path, and ``..`` / absolute paths
    cannot escape the sources folder (checked after resolving symlinks).
    """
    if not filename or len(filename) > 255 or "\x00" in filename:
        raise InvalidInputError("Please provide a file name from the notes://sources list.")
    candidate = Path(filename)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise InvalidInputError(
            "Only files inside the configured sources folder can be ingested. "
            "Read the notes://sources resource to see what is available."
        )
    path = (sources_dir / candidate).resolve()
    if not path.is_relative_to(sources_dir.resolve()):
        raise InvalidInputError("Only files inside the configured sources folder can be ingested.")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InvalidInputError(
            f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )
    if not path.is_file():
        raise NotFoundError(
            f"No document named '{candidate.name}' in the sources folder. "
            "Read notes://sources for the list of files."
        )
    if path.stat().st_size > max_mb * 1024 * 1024:
        raise InvalidInputError(f"File is larger than the {max_mb} MB limit.")
    return path


def extract_chunks(path: Path) -> list[Chunk]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pptx":
            chunks = _from_pptx(path)
        elif suffix == ".pdf":
            chunks = _from_pdf(path)
        else:
            chunks = _from_markdown(path.read_text(encoding="utf-8", errors="replace"), path.stem)
    except Exception as exc:  # parser libraries raise many different exception types
        raise DocumentParseError(
            f"Could not read '{path.name}'. The file may be corrupted or password-protected.",
            detail=f"{type(exc).__name__}: {exc}",  # often contains the absolute path - log only
        ) from exc
    return [c for c in chunks if c.text.strip()]


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()[:MAX_CHUNK_CHARS]


def _from_pptx(path: Path) -> list[Chunk]:
    from pptx import Presentation  # imported lazily: only needed for decks

    chunks = []
    for number, slide in enumerate(Presentation(str(path)).slides, start=1):
        title = slide.shapes.title.text_frame.text.strip() if slide.shapes.title else ""
        lines = [
            shape.text_frame.text
            for shape in slide.shapes
            if shape.has_text_frame and shape != slide.shapes.title
        ]
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                lines.append(f"Speaker notes: {notes}")
        chunks.append(Chunk(title or f"{path.stem} - slide {number}", _clean("\n".join(lines)), f"slide {number}"))
    return chunks


def _from_pdf(path: Path) -> list[Chunk]:
    from pypdf import PdfReader

    chunks = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        title = first_line[:80] or f"{path.stem} - page {number}"
        chunks.append(Chunk(title, _clean(text), f"page {number}"))
    return chunks


def _from_markdown(text: str, fallback_title: str) -> list[Chunk]:
    """Split on Markdown headings; text before the first heading becomes its own chunk."""
    sections: list[tuple[str, list[str]]] = [(fallback_title, [])]
    for line in text.splitlines():
        heading = re.match(r"^#{1,3}\s+(.*)", line)
        if heading:
            sections.append((heading.group(1).strip(), []))
        else:
            sections[-1][1].append(line)
    return [
        Chunk(title, _clean("\n".join(body)), f"section {i}")
        for i, (title, body) in enumerate(sections)
        if "\n".join(body).strip()
    ]
