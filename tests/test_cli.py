"""CLI ingest and query against the offline providers."""

from pathlib import Path

from app.cli import main
from app.config import get_settings


def test_cli_ingest_query_and_documents(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "index"))
    get_settings.cache_clear()
    handbook = tmp_path / "handbook.md"
    handbook.write_text(
        "# Guide\n\n## Refunds\n\nThe refund window is 30 days from the purchase date.\n",
        encoding="utf-8",
    )

    assert main(["ingest", str(handbook)]) == 0
    ingested = capsys.readouterr().out
    assert "handbook.md" in ingested

    assert main(["documents"]) == 0
    listed = capsys.readouterr().out
    assert "handbook.md" in listed

    assert main(["query", "How long is the refund window?"]) == 0
    answer = capsys.readouterr().out
    assert "30 days" in answer
    assert "handbook.md, chunk" in answer
    assert "[1]" in answer
    get_settings.cache_clear()


def test_cli_missing_file_is_an_error(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "index"))
    get_settings.cache_clear()
    assert main(["ingest", str(tmp_path / "missing.md")]) == 1
    assert "File not found" in capsys.readouterr().err
    get_settings.cache_clear()
