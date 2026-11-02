"""HTTP API coverage for the offline providers."""

import io
from pathlib import Path

from docx import Document
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.citations import NOT_FOUND_ANSWER
from app.services.errors import ProviderConfigError, ProviderError
from tests.conftest import build_settings
from tests.pdfutil import make_pdf

ROOT = Path(__file__).resolve().parents[1]
HANDBOOK = (ROOT / "examples" / "handbook.md").read_bytes()


def test_root_and_health(client: TestClient) -> None:
    root = client.get("/")
    health = client.get("/health")

    assert root.status_code == 200
    assert root.json()["status"] == "ok"
    assert root.json()["embedding_provider"] == "offline"
    assert root.json()["llm_provider"] == "offline"
    assert root.json()["vector_store"] == "local"
    assert root.json()["docs"] == "/docs"
    assert root.json()["ui"] == "/ui"
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "embedding_provider": "offline",
        "llm_provider": "offline",
        "vector_store": "local",
        "documents": 0,
        "chunks": 0,
    }


def test_ui_is_served(client: TestClient) -> None:
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "RAG Knowledge Base" in response.text
    assert "/query" in response.text


def test_ingest_and_query_handbook_cites_every_claim(client: TestClient) -> None:
    ingested = client.post(
        "/ingest",
        files={"files": ("handbook.md", HANDBOOK, "text/markdown")},
    )
    assert ingested.status_code == 200
    body = ingested.json()
    assert body["documents"][0]["source"] == "handbook.md"
    assert body["documents"][0]["chunks"] >= 3
    assert body["embedding_provider"] == "offline"

    response = client.post("/query", json={"question": "How long is the refund window?"})
    assert response.status_code == 200
    answer = response.json()
    assert answer["refused"] is False
    assert answer["provider"] == "offline"
    assert "30 days" in answer["answer"]
    assert answer["claims"]
    assert answer["citations"]
    for claim in answer["claims"]:
        assert claim["citations"], claim
        for citation in claim["citations"]:
            assert citation["source"] == "handbook.md"
            assert citation["page"] is None
            assert citation["label"].startswith("handbook.md, chunk ")
            assert "chunk" in citation["label"]
    markers = [f"[{index}]" for index in range(1, len(answer["citations"]) + 1)]
    for marker in markers:
        assert marker in answer["answer"]


def test_query_with_an_empty_index_refuses(client: TestClient) -> None:
    response = client.post("/query", json={"question": "How long is the refund window?"})
    assert response.status_code == 200
    assert response.json()["refused"] is True
    assert response.json()["answer"] == NOT_FOUND_ANSWER
    assert response.json()["claims"] == []
    assert response.json()["citations"] == []


def test_pdf_citation_includes_the_page(client: TestClient) -> None:
    pdf = make_pdf(
        [
            "The refund window is 30 days from the purchase date.",
            "Support hours are weekdays from 09:00 to 17:00 WAT.",
        ]
    )
    ingested = client.post("/ingest", files={"files": ("policy.pdf", pdf, "application/pdf")})
    assert ingested.status_code == 200
    assert ingested.json()["documents"][0]["pages"] == 2

    response = client.post("/query", json={"question": "What are the support hours?"})
    citation = response.json()["claims"][0]["citations"][0]
    assert citation["source"] == "policy.pdf"
    assert citation["page"] == 2
    assert citation["label"] == "policy.pdf, page 2, chunk 2"


def test_docx_ingest(client: TestClient) -> None:
    document = Document()
    document.add_paragraph("Rail is the default for staff travel to Abuja.")
    buffer = io.BytesIO()
    document.save(buffer)
    ingested = client.post(
        "/ingest",
        files={
            "files": (
                "travel.docx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert ingested.status_code == 200
    response = client.post(
        "/query",
        json={"question": "What is the default for staff travel to Abuja?"},
    )
    assert response.status_code == 200
    assert response.json()["claims"][0]["citations"][0]["source"] == "travel.docx"
    assert "Abuja" in response.json()["answer"]


def test_text_endpoint_lists_and_deletes(client: TestClient) -> None:
    created = client.post(
        "/documents",
        json={
            "filename": "notes.txt",
            "text": "The refund window is 30 days from the purchase date.",
        },
    )
    assert created.status_code == 200
    document_id = created.json()["documents"][0]["document_id"]

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert listed.json()["documents"] == [
        {"document_id": document_id, "source": "notes.txt", "chunks": 1}
    ]

    health = client.get("/health")
    assert health.json()["documents"] == 1
    assert health.json()["chunks"] == 1

    removed = client.delete(f"/documents/{document_id}")
    assert removed.status_code == 200
    assert removed.json()["chunks_removed"] == 1
    assert client.get("/documents").json()["documents"] == []
    missing = client.delete(f"/documents/{document_id}")
    assert missing.status_code == 404
    assert client.delete("/documents/not-an-id").status_code == 404


def test_rejects_unsupported_empty_and_oversized_uploads(client: TestClient) -> None:
    unsupported = client.post("/ingest", files={"files": ("picture.png", b"\x89PNG", "image/png")})
    assert unsupported.status_code == 400
    assert "Unsupported" in unsupported.json()["detail"]

    empty = client.post("/ingest", files={"files": ("empty.txt", b"", "text/plain")})
    assert empty.status_code == 400
    assert empty.json()["detail"] == "Upload is empty."

    blank = client.post("/query", json={"question": "   "})
    assert blank.status_code == 422
    assert client.post("/query", json={"question": "refunds", "top_k": 0}).status_code == 422


def test_upload_over_limit_is_rejected(tmp_path: Path) -> None:
    app = create_app(build_settings(data_dir=tmp_path, max_upload_bytes=8))
    with TestClient(app) as client:
        response = client.post(
            "/ingest",
            files={"files": ("notes.txt", b"123456789", "text/plain")},
        )
    assert response.status_code == 413


def test_provider_misconfiguration_is_unavailable(tmp_path: Path) -> None:
    app = create_app(build_settings(data_dir=tmp_path))

    class Missing:
        name = "missing"

        def answer(self, question: str, chunks: list[object]) -> None:
            raise ProviderConfigError("provider is not configured")

    app.state.knowledge_base.answerer = Missing()
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "How long is the refund window?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "provider is not configured"


def test_provider_failure_is_bad_gateway(tmp_path: Path) -> None:
    app = create_app(build_settings(data_dir=tmp_path))

    class Failing:
        name = "broken"

        def answer(self, question: str, chunks: list[object]) -> None:
            raise ProviderError("upstream down")

    app.state.knowledge_base.answerer = Failing()
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "How long is the refund window?"})
    assert response.status_code == 502
    assert response.json()["detail"] == "upstream down"
