"""Readers for text, Markdown, PDF, and DOCX."""

import io

import pytest
from docx import Document

from app.services.documents.loaders import load_bytes
from app.services.errors import EmptyDocumentError, InvalidDocumentError, UnsupportedDocumentError
from tests.pdfutil import make_pdf


def test_markdown_splits_on_headings() -> None:
    loaded = load_bytes(
        "guide.md",
        b"# Guide\n\n## Refunds\n\nThe refund window is 30 days.\n",
    )
    assert loaded.source == "guide.md"
    assert [page.number for page in loaded.pages] == [None, None]
    assert "Refunds" in loaded.pages[1].text


def test_text_file_is_one_section_and_paths_are_stripped() -> None:
    loaded = load_bytes("../secret/notes.txt", b"Hello from the notes.\n")
    assert loaded.source == "notes.txt"
    assert loaded.pages[0].text == "Hello from the notes.\n"


def test_pdf_pages_keep_numbers() -> None:
    data = make_pdf(
        [
            "The refund window is 30 days from the purchase date.",
            "Support hours are weekdays from 09:00 to 17:00 WAT.",
        ]
    )
    loaded = load_bytes("policy.pdf", data)
    assert [page.number for page in loaded.pages] == [1, 2]
    assert "30 days" in loaded.pages[0].text
    assert "09:00" in loaded.pages[1].text


def test_docx_reads_paragraphs_and_table_cells() -> None:
    document = Document()
    document.add_paragraph("Rail is the default for staff travel to Abuja.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "City"
    table.rows[0].cells[1].text = "Lagos"
    buffer = io.BytesIO()
    document.save(buffer)

    loaded = load_bytes("travel.docx", buffer.getvalue())
    assert loaded.pages[0].number is None
    assert "Abuja" in loaded.pages[0].text
    assert "City | Lagos" in loaded.pages[0].text


def test_unsupported_empty_and_corrupt_files_are_rejected() -> None:
    with pytest.raises(UnsupportedDocumentError, match="Unsupported file type"):
        load_bytes("picture.png", b"\x89PNG")
    with pytest.raises(EmptyDocumentError):
        load_bytes("empty.txt", b"")
    with pytest.raises(InvalidDocumentError, match="PDF"):
        load_bytes("broken.pdf", b"not a pdf")
    with pytest.raises(InvalidDocumentError, match="DOCX"):
        load_bytes("broken.docx", b"not a docx")
    with pytest.raises(UnsupportedDocumentError, match="file name"):
        load_bytes("", b"hello")
