"""Shared test fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def build_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "embedding_provider": "offline",
        "llm_provider": "offline",
        "vector_store": "local",
        "openai_api_key": None,
        "data_dir": Path("./data"),
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(build_settings(data_dir=tmp_path / "index"))
    with TestClient(app) as test_client:
        yield test_client
