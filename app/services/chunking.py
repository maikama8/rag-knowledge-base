"""Split document text into chunks that keep a page or section identity."""

import re

from app.services.models import DocumentChunk, PageContent


def chunk_pages(
    pages: tuple[PageContent, ...] | list[PageContent],
    *,
    source: str,
    document_id: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """Chunk each page separately so a citation can name a single page."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be zero or smaller than chunk_size.")

    chunks: list[DocumentChunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size=chunk_size, overlap=overlap):
            index = len(chunks)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}:{index}",
                    document_id=document_id,
                    source=source,
                    text=piece,
                    chunk_index=index,
                    page=page.number,
                )
            )
    return chunks


def split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Pack paragraphs up to `chunk_size`. Oversized paragraphs slide forward with overlap."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be zero or smaller than chunk_size.")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_window(paragraph, chunk_size=chunk_size, overlap=overlap))
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = paragraph
    if current:
        pieces.append(current)
    return pieces


def split_markdown_sections(text: str) -> list[str]:
    """Split Markdown on headings so each section can be cited on its own."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    sections: list[list[str]] = []
    current: list[str] = []
    for line in normalized.split("\n"):
        if line.startswith("#") and any(part.strip() for part in current):
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if any(part.strip() for part in current):
        sections.append(current)

    return ["\n".join(section).strip() for section in sections if section]


def _window(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        if end < length:
            snap = text.rfind(" ", start + (chunk_size // 2), end)
            if snap != -1:
                end = snap
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        next_start = end - overlap if overlap else end
        if next_start <= start:
            next_start = end if end > start else start + 1
        start = next_start
    return pieces
