"""Read PDF, Markdown, text, and DOCX uploads into pages."""

import hashlib
import io
import zipfile
from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.services.chunking import split_markdown_sections
from app.services.errors import EmptyDocumentError, InvalidDocumentError, UnsupportedDocumentError
from app.services.models import LoadedDocument, PageContent

_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def safe_source_name(filename: str | None) -> str:
    """Keep the base name so an upload cannot carry a directory path."""
    name = Path(filename or "").name.strip()
    if not name or name in {".", ".."}:
        raise UnsupportedDocumentError("The upload needs a file name.")
    return name


def load_bytes(filename: str | None, data: bytes) -> LoadedDocument:
    """Load an uploaded file. PDF pages keep their numbers. Markdown splits on headings."""
    source = safe_source_name(filename)
    suffix = Path(source).suffix.lower()
    if suffix in {".md", ".markdown"}:
        pages = _markdown_pages(_decode_text(data, source))
    elif suffix == ".txt":
        pages = (PageContent(number=None, text=_decode_text(data, source)),)
    elif suffix == ".pdf":
        pages = tuple(_pdf_pages(data, source))
    elif suffix == ".docx":
        pages = (PageContent(number=None, text=_docx_text(data, source)),)
    else:
        kind = suffix or source
        raise UnsupportedDocumentError(
            f"Unsupported file type '{kind}'. Upload a PDF, Markdown, TXT, or DOCX file."
        )
    if not any(page.text.strip() for page in pages):
        raise EmptyDocumentError(f"{source} has no extractable text.")
    return LoadedDocument(source=source, pages=pages)


def load_text(filename: str, text: str) -> LoadedDocument:
    """Load a raw string using the same rules as a text or Markdown upload."""
    source = safe_source_name(filename)
    encoded = text.encode("utf-8")
    return load_bytes(source, encoded)


def document_id_for(source: str) -> str:
    """Stable id for a file name. Re-ingesting the same name replaces that file."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _decode_text(data: bytes, source: str) -> str:
    if not data:
        raise EmptyDocumentError(f"{source} is empty.")
    if b"\x00" in data:
        raise InvalidDocumentError(f"{source} is not valid UTF-8 text.")
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise InvalidDocumentError(f"{source} is not valid UTF-8 text.")


def _markdown_pages(text: str) -> tuple[PageContent, ...]:
    sections = split_markdown_sections(text)
    return tuple(PageContent(number=None, text=section) for section in sections)


def _pdf_pages(data: bytes, source: str) -> list[PageContent]:
    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, OSError, ValueError) as exc:
        raise InvalidDocumentError(f"{source} could not be read as a PDF.") from exc
    if getattr(reader, "is_encrypted", False):
        try:
            unlocked = reader.decrypt("")
        except (PdfReadError, OSError, ValueError, NotImplementedError) as exc:
            raise InvalidDocumentError(f"{source} is encrypted and could not be read.") from exc
        if unlocked == 0:
            raise InvalidDocumentError(f"{source} is encrypted and could not be read.")
    pages: list[PageContent] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            extracted = page.extract_text() or ""
        except (PdfReadError, OSError, ValueError, KeyError) as exc:
            raise InvalidDocumentError(f"{source} page {number} could not be read.") from exc
        pages.append(PageContent(number=number, text=extracted))
    return pages


def _docx_text(data: bytes, source: str) -> str:
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise InvalidDocumentError(f"{source} could not be read as a DOCX file.")
    try:
        document = Document(io.BytesIO(data))
    except (PackageNotFoundError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise InvalidDocumentError(f"{source} could not be read as a DOCX file.") from exc
    parts: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)
    for table in document.tables:
        seen: set[str] = set()
        for row in table.rows:
            cells: list[str] = []
            for cell in row.cells:
                value = cell.text.strip()
                if value and value not in cells:
                    cells.append(value)
            line = " | ".join(cells)
            if line and line not in seen:
                seen.add(line)
                parts.append(line)
    return "\n\n".join(parts)
