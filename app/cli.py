"""Command-line ingest and question answering."""

import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.services.errors import (
    EmptyDocumentError,
    InvalidDocumentError,
    ProviderConfigError,
    ProviderError,
    UnsupportedDocumentError,
    VectorStoreError,
)
from app.services.factory import build_knowledge_base
from app.services.models import Answer


def main(argv: list[str] | None = None) -> int:
    """Run `ingest`, `query`, `documents`, or `serve`."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            return _ingest(args.paths)
        if args.command == "query":
            return _query(args.question, args.top_k)
        if args.command == "documents":
            return _documents()
        if args.command == "serve":
            return _serve(args.host, args.port)
    except (
        UnsupportedDocumentError,
        InvalidDocumentError,
        EmptyDocumentError,
        VectorStoreError,
        ProviderConfigError,
        ProviderError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.error(f"Unknown command: {args.command}")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Index documents and ask questions with citations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index PDF, Markdown, TXT, or DOCX files")
    ingest.add_argument("paths", nargs="+", help="Files to index")

    query = sub.add_parser("query", help="Ask a question against the local index")
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=None, dest="top_k")

    sub.add_parser("documents", help="List indexed files")

    serve = sub.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _ingest(paths: list[str]) -> int:
    payloads: list[tuple[str, bytes]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            return 1
        payloads.append((path.name, path.read_bytes()))
    knowledge_base = build_knowledge_base(get_settings())
    for item in knowledge_base.ingest_files(payloads):
        print(f"indexed {item.chunks} chunks from {item.source} ({item.document_id})")
    return 0


def _query(question: str, top_k: int | None) -> int:
    if top_k is not None and not 1 <= top_k <= 20:
        print("--top-k must be between 1 and 20.", file=sys.stderr)
        return 1
    answer = build_knowledge_base(get_settings()).query(question, top_k)
    _print_answer(answer)
    return 0


def _documents() -> int:
    documents = build_knowledge_base(get_settings()).documents()
    if not documents:
        print("No documents indexed.")
        return 0
    for document in documents:
        label = "chunk" if document.chunks == 1 else "chunks"
        print(f"{document.document_id}  {document.chunks} {label}  {document.source}")
    return 0


def _serve(host: str, port: int) -> int:
    import uvicorn

    uvicorn.run("app.main:app", host=host, port=port, reload=False)
    return 0


def _print_answer(answer: Answer) -> None:
    print(answer.answer)
    if not answer.citations:
        return
    print()
    for number, citation in enumerate(answer.citations, start=1):
        print(f"[{number}] {citation.label} (score {citation.score:.4f})")
        print(f"    {citation.excerpt}")


if __name__ == "__main__":
    raise SystemExit(main())
