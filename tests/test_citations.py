"""Citation labels and the rule that every claim is cited."""

import pytest

from app.services.citations import (
    NOT_FOUND_ANSWER,
    build_answer,
    citation_from_chunk,
    format_citation,
)
from app.services.errors import UncitedClaimError
from app.services.models import Claim, DocumentChunk
from app.services.text import split_sentences


def _chunk(**overrides: object) -> DocumentChunk:
    values: dict[str, object] = {
        "chunk_id": "doc:0",
        "document_id": "doc",
        "source": "notes.md",
        "text": "The refund window is 30 days from the purchase date.",
        "chunk_index": 0,
        "page": None,
    }
    values.update(overrides)
    return DocumentChunk(**values)  # type: ignore[arg-type]


def test_markdown_headings_are_not_part_of_the_sentence() -> None:
    text = (
        "## Refunds\n\nThe refund window is 30 days from the purchase date. Refunds are returned."
    )
    assert split_sentences(text) == [
        "The refund window is 30 days from the purchase date.",
        "Refunds are returned.",
    ]


def test_excerpt_skips_markdown_headings() -> None:
    citation = citation_from_chunk(
        _chunk(text="## Refunds\n\nThe refund window is 30 days from the purchase date."),
        0.5,
    )
    assert citation.excerpt == "The refund window is 30 days from the purchase date."


def test_format_citation_with_a_page() -> None:
    assert format_citation("policy.pdf", page=3, chunk_index=1) == "policy.pdf, page 3, chunk 2"


def test_format_citation_without_a_page() -> None:
    assert format_citation("notes.md", page=None, chunk_index=0) == "notes.md, chunk 1"


def test_answer_marks_every_claim_and_lists_the_same_sources() -> None:
    first = citation_from_chunk(_chunk(), 0.81234)
    second = citation_from_chunk(
        _chunk(
            chunk_id="doc:1",
            chunk_index=1,
            page=4,
            source="policy.pdf",
            text="Support hours are weekdays from 09:00 to 17:00 WAT.",
        ),
        0.5,
    )
    answer = build_answer(
        question="refunds and hours",
        provider="offline",
        claims=[
            Claim(text="The refund window is 30 days from the purchase date.", citations=(first,)),
            Claim(text="Support hours are weekdays.", citations=(first, second)),
        ],
    )

    assert answer.refused is False
    assert answer.answer == (
        "The refund window is 30 days from the purchase date. [1] "
        "Support hours are weekdays. [1][2]"
    )
    assert [citation.label for citation in answer.citations] == [
        "notes.md, chunk 1",
        "policy.pdf, page 4, chunk 2",
    ]
    assert answer.citations[0].score == 0.8123
    assert answer.claims[0].citations[0].source == "notes.md"
    assert answer.claims[1].citations[1].page == 4
    for claim in answer.claims:
        assert claim.citations
        assert claim.text.rstrip(".") in answer.answer


def test_missing_claims_are_a_refusal() -> None:
    answer = build_answer(question="wifi password", claims=[], provider="offline")
    assert answer.refused is True
    assert answer.answer == NOT_FOUND_ANSWER
    assert answer.claims == ()
    assert answer.citations == ()


def test_uncited_claim_is_rejected() -> None:
    with pytest.raises(UncitedClaimError, match="citation"):
        build_answer(
            question="refunds",
            claims=[Claim(text="The refund window is 30 days.", citations=())],
            provider="offline",
        )
