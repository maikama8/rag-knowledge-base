"""Provider selection and the OpenAI HTTP adapters."""

import json

import httpx
import pytest
from pydantic import ValidationError

from app.main import create_app
from app.services.errors import ProviderConfigError, ProviderError
from app.services.factory import (
    build_answer_provider,
    build_embedding_provider,
    build_knowledge_base,
)
from app.services.llm.openai import OpenAIAnswerProvider
from app.services.models import DocumentChunk, ScoredChunk
from tests.conftest import build_settings


def test_factories_default_to_offline_providers(tmp_path) -> None:
    settings = build_settings(data_dir=tmp_path)
    assert build_embedding_provider(settings).name == "offline"
    assert build_answer_provider(settings).name == "offline"
    knowledge = build_knowledge_base(settings)
    assert knowledge.store.name == "local"
    assert knowledge.embeddings.dimension == 512


def test_openai_provider_requires_an_api_key(tmp_path) -> None:
    settings = build_settings(
        data_dir=tmp_path,
        embedding_provider="openai",
        llm_provider="offline",
        openai_api_key=None,
    )
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        build_embedding_provider(settings)
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        create_app(settings)

    chat = build_settings(data_dir=tmp_path, llm_provider="openai", openai_api_key="  ")
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        build_answer_provider(chat)


def test_provider_name_is_normalized(tmp_path) -> None:
    settings = build_settings(
        data_dir=tmp_path,
        embedding_provider=" OpenAI ",
        llm_provider=" OpenAI ",
        openai_api_key="sk-test",
    )
    assert settings.embedding_provider == "openai"
    assert settings.llm_provider == "openai"
    assert build_embedding_provider(settings).name == "openai"
    assert build_answer_provider(settings).name == "openai"


def test_unknown_provider_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError):
        build_settings(data_dir=tmp_path, embedding_provider="nope")


def test_overlap_cannot_exceed_chunk_size(tmp_path) -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        build_settings(data_dir=tmp_path, chunk_size=200, chunk_overlap=200)


def test_openai_embeddings_parse_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def post(self, url: str, **kwargs: object) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = kwargs["headers"]
            captured["json"] = kwargs["json"]
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    from app.services.embeddings.openai import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        base_url="https://api.openai.com/v1/",
        model="text-embedding-3-small",
        timeout=5,
        dimension=2,
    )
    vectors = provider.embed(["refunds", "hours"])

    assert captured["url"] == "https://api.openai.com/v1/embeddings"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer sk-test"
    body = captured["json"]
    assert isinstance(body, dict)
    assert body["model"] == "text-embedding-3-small"
    assert body["dimensions"] == 2
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


def test_openai_answerer_drops_uncited_and_unknown_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = DocumentChunk(
        chunk_id="doc:0",
        document_id="doc",
        source="handbook.md",
        text="The refund window is 30 days from the purchase date.",
        chunk_index=0,
        page=None,
    )
    hits = [ScoredChunk(chunk=chunk, score=0.66)]

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            content = {
                "unknown": False,
                "claims": [
                    {
                        "text": "The refund window is 30 days from the purchase date.",
                        "chunk_ids": ["doc:0", "missing"],
                    },
                    {"text": "Invented office hours.", "chunk_ids": ["nope"]},
                    {"text": "No ids at all.", "chunk_ids": []},
                ],
            }
            fenced = "```json\n" + json.dumps(content) + "\n```"
            return {"choices": [{"message": {"content": fenced}}]}

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def post(self, url: str, **kwargs: object) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    provider = OpenAIAnswerProvider(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        timeout=5,
    )
    answer = provider.answer("How long is the refund window?", hits)

    assert answer.provider == "openai"
    assert answer.refused is False
    assert len(answer.claims) == 1
    assert answer.claims[0].citations[0].chunk_id == "doc:0"
    assert answer.claims[0].citations[0].label == "handbook.md, chunk 1"
    assert answer.citations[0].source == "handbook.md"
    assert "[1]" in answer.answer
    assert "Invented" not in answer.answer
    assert "sk-test" not in answer.answer


def test_openai_invalid_json_is_a_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = DocumentChunk(
        chunk_id="doc:0",
        document_id="doc",
        source="notes.md",
        text="Some text.",
        chunk_index=0,
        page=None,
    )

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "not-json"}}]}

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def post(self, url: str, **kwargs: object) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    provider = OpenAIAnswerProvider(
        api_key="sk-test",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
        timeout=5,
    )
    with pytest.raises(ProviderError, match="valid JSON"):
        provider.answer("question", [ScoredChunk(chunk=chunk, score=0.4)])
