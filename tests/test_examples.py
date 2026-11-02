"""The checked handbook query stays in sync with examples/sample_query.json."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def test_handbook_query_matches_checked_output(client: TestClient) -> None:
    handbook = (ROOT / "examples" / "handbook.md").read_bytes()
    ingested = client.post(
        "/ingest",
        files={"files": ("handbook.md", handbook, "text/markdown")},
    )
    assert ingested.status_code == 200

    response = client.post("/query", json={"question": "How long is the refund window?"})
    expected = json.loads((ROOT / "examples" / "sample_query.json").read_text(encoding="utf-8"))

    assert response.status_code == 200
    assert response.json() == expected
