from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ocha.api.main import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The real model is 14.2 GB, so the suite runs with the load skipped."""
    monkeypatch.setenv("OCHA_SKIP_MODEL", "1")
    monkeypatch.setattr("ocha.api.main.SKIP_MODEL", True)
    with TestClient(app) as c:
        yield c


def test_health_reports_model_and_memory(client: TestClient) -> None:
    """T1.7 acceptance: /health reports model loaded and resident memory."""
    body = client.get("/health").json()
    assert set(body) == {
        "status",
        "model",
        "model_loaded",
        "resident_memory_gb",
        "grammar_entries",
    }
    assert body["model"] == "mlx-community/Qwen3.5-9B-4bit"  # DECISION.md
    assert body["grammar_entries"] == 20


def test_skipped_load_is_reported_as_degraded_not_ok(client: TestClient) -> None:
    """A skipped load must never look like a working one -- that would be exactly
    the silent fallback T1.7 forbids."""
    body = client.get("/health").json()
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"


def test_grammar_reference_is_loaded_at_startup(client: TestClient) -> None:
    """Fail-fast: a malformed reference means the firewall has nothing to serve."""
    assert app.state.grammar is not None
    assert len(app.state.grammar) == 20


def test_the_client_is_served_from_the_same_origin(client: TestClient) -> None:
    """The client resolves its WebSocket URL from `location.host`.

    Served from anywhere else it would have to be told where the API is, and the
    phone would need two hostnames instead of one. The mount is registered last on
    purpose: a mount at "/" matches everything, and Starlette checks routes in
    registration order, so mounted earlier it silently shadows /health and /turn.
    """
    page = client.get("/")
    assert page.status_code == 200
    assert "AudioWorklet" in page.text, "the served page is not the voice client"
    # The routes the mount would have swallowed.
    assert client.get("/health").status_code == 200
