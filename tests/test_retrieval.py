"""Ranking, replacement, and persistence of the local vector store."""

from pathlib import Path

import pytest

from app.services.embeddings.offline import OfflineEmbeddingProvider
from app.services.errors import VectorStoreError
from app.services.factory import build_knowledge_base
from app.services.vectorstore.local import LocalVectorStore
from tests.conftest import build_settings

REFUND = "The refund window is 30 days from the purchase date."
HOURS = "Support hours are weekdays from 09:00 to 17:00 WAT."


def test_matching_document_ranks_first(tmp_path: Path) -> None:
    knowledge = build_knowledge_base(
        build_settings(data_dir=tmp_path / "kb", offline_embedding_dimensions=128, min_score=0.15)
    )
    knowledge.ingest_text("refunds.md", REFUND)
    knowledge.ingest_text("hours.md", HOURS)

    hits = knowledge.store.query(knowledge.embeddings.embed(["refund window"])[0], top_k=2)
    assert hits[0].chunk.source == "refunds.md"
    assert hits[0].score > hits[1].score
    assert hits[1].chunk.source == "hours.md"

    answer = knowledge.query("How long is the refund window?")
    assert answer.refused is False
    assert answer.claims[0].citations[0].source == "refunds.md"
    assert "30 days" in answer.answer

    hours = knowledge.query("What are the support hours?")
    assert hours.claims[0].citations[0].source == "hours.md"
    assert "09:00" in hours.answer


def test_unrelated_question_is_refused(tmp_path: Path) -> None:
    knowledge = build_knowledge_base(build_settings(data_dir=tmp_path))
    knowledge.ingest_text("refunds.md", REFUND)
    answer = knowledge.query("What is the office wifi password?")
    assert answer.refused is True
    assert answer.claims == ()
    assert answer.citations == ()


def test_reingest_replaces_the_same_file_name(tmp_path: Path) -> None:
    knowledge = build_knowledge_base(build_settings(data_dir=tmp_path))
    first = knowledge.ingest_text("notes.md", REFUND)
    second = knowledge.ingest_text("notes.md", HOURS)
    assert first.document_id == second.document_id
    assert knowledge.store.count() == second.chunks
    assert len(knowledge.documents()) == 1
    answer = knowledge.query("What are the support hours?")
    assert answer.claims[0].citations[0].source == "notes.md"
    refund = knowledge.query("How long is the refund window?")
    assert refund.refused is True


def test_delete_removes_the_document_from_ranking(tmp_path: Path) -> None:
    knowledge = build_knowledge_base(build_settings(data_dir=tmp_path))
    ingested = knowledge.ingest_text("refunds.md", REFUND)
    knowledge.ingest_text("hours.md", HOURS)
    removed = knowledge.delete(ingested.document_id)
    assert removed >= 1
    assert knowledge.delete(ingested.document_id) == 0
    answer = knowledge.query("How long is the refund window?")
    assert answer.refused is True
    assert [item.source for item in knowledge.documents()] == ["hours.md"]


def test_index_reloads_from_disk(tmp_path: Path) -> None:
    embeddings = OfflineEmbeddingProvider(dimension=128)
    knowledge = build_knowledge_base(
        build_settings(data_dir=tmp_path, offline_embedding_dimensions=embeddings.dimension)
    )
    knowledge.ingest_text("refunds.md", REFUND)

    reloaded = LocalVectorStore(
        tmp_path,
        embedding_provider="offline",
        dimension=embeddings.dimension,
    )
    hits = reloaded.query(embeddings.embed(["refund window"])[0], top_k=1)
    assert reloaded.count() == 1
    assert hits[0].chunk.source == "refunds.md"
    assert "30 days" in hits[0].chunk.text


def test_index_rejects_a_different_embedding_provider(tmp_path: Path) -> None:
    knowledge = build_knowledge_base(build_settings(data_dir=tmp_path))
    knowledge.ingest_text("refunds.md", REFUND)
    with pytest.raises(VectorStoreError, match="different embedding provider"):
        LocalVectorStore(tmp_path, embedding_provider="openai", dimension=512)
