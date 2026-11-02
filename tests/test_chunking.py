"""Chunk boundaries, overlap, and page numbers."""

import pytest

from app.services.chunking import chunk_pages, split_markdown_sections, split_text
from app.services.models import PageContent


def test_short_text_is_one_chunk() -> None:
    pieces = split_text("The refund window is 30 days.", chunk_size=200, overlap=20)
    assert pieces == ["The refund window is 30 days."]


def test_blank_text_yields_nothing() -> None:
    assert split_text("  \n\n  ", chunk_size=200, overlap=20) == []
    chunks = chunk_pages(
        [PageContent(number=1, text="   ")],
        source="empty.pdf",
        document_id="doc",
        chunk_size=200,
        overlap=20,
    )
    assert chunks == []


def test_paragraphs_pack_until_the_size_limit() -> None:
    first = "A" * 40
    second = "B" * 40
    third = "C" * 40
    pieces = split_text(f"{first}\n\n{second}\n\n{third}", chunk_size=90, overlap=10)
    assert pieces[0] == f"{first}\n\n{second}"
    assert pieces[1] == third
    assert first not in pieces[1]


def test_long_paragraph_overlaps_the_next_window() -> None:
    words = [f"w{index:03d}" for index in range(80)]
    pieces = split_text(" ".join(words), chunk_size=80, overlap=20)
    assert len(pieces) >= 3
    last_word = pieces[0].split()[-1]
    assert last_word in pieces[1].split()
    for piece in pieces:
        assert piece.strip()


def test_overlap_must_be_smaller_than_the_chunk() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_text("hello world", chunk_size=20, overlap=20)


def test_markdown_sections_follow_headings() -> None:
    text = "# Guide\n\n## Refunds\n\nThirty days.\n\n## Support\n\nWeekdays.\n"
    sections = split_markdown_sections(text)
    assert sections == ["# Guide", "## Refunds\n\nThirty days.", "## Support\n\nWeekdays."]


def test_pages_keep_their_numbers_and_chunk_ids_increase() -> None:
    chunks = chunk_pages(
        [
            PageContent(number=1, text="Page one is about refunds."),
            PageContent(number=2, text="Page two is about support hours."),
        ],
        source="policy.pdf",
        document_id="abc",
        chunk_size=200,
        overlap=20,
    )
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.chunk_id for chunk in chunks] == ["abc:0", "abc:1"]
    assert chunks[0].source == "policy.pdf"
    assert len({chunk.chunk_id for chunk in chunks}) == 2
