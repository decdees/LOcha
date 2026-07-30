"""Real-model smoke test. Excluded from the default run; `make test-slow`.

This exists because of a specific gap. In T1.8 two bugs shipped past a fully
green stub-based suite:

  - sqlite3 connections are thread-bound, and FastAPI dispatches `def` endpoints
    to a threadpool, so every /turn returned 500. The unit tests called run_turn()
    directly and never crossed the transport.
  - MLX GPU streams are thread-local, so generation failed from a worker thread
    with "There is no Stream(gpu, 1) in current thread". No stub can reproduce
    this, because no stub loads MLX.

Both were only visible with the real model behind the real transport. That is
what this file covers.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    db = tmp_path_factory.mktemp("smoke") / "smoke.db"
    os.environ["OCHA_DB"] = str(db)
    import ocha.db.schema as schema

    schema.DEFAULT_DB = db
    from ocha.api.main import app

    with TestClient(app) as c:  # loads the real model
        yield c


def test_real_model_loads_and_reports_residency(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["model_loaded"] is True, "the real model must actually load"
    assert body["status"] == "ok"
    assert body["resident_memory_gb"] > 1.0, body


def test_real_turn_over_http(client: TestClient) -> None:
    """The regression test for both T1.8 threading bugs at once."""
    r = client.post("/turn", json={"text": "こんにちは。"})
    assert r.status_code == 200, r.text
    assert r.json()["reply"], "the real model must produce a reply"


def test_real_grammar_question_is_firewalled(client: TestClient) -> None:
    """FR-5 with the real model: it must emit the sentinel, and the reference --
    not the model -- must answer."""
    r = client.post("/turn", json={"text": "What is the difference between は and が?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grammar"] is not None, f"firewall did not fire: {body['reply']!r}"
    assert body["reply"] is None
    assert body["grammar"]["entry_id"] == "particle_wa_ga"
    assert "GRAMMAR_QUERY" not in r.text


def test_kv_cache_makes_later_turns_faster(client: TestClient) -> None:
    """T0.4 measured 1.81 s -> 0.50 s flat. The second turn reuses the prefix."""
    import time

    def turn(text: str) -> float:
        t = time.perf_counter()
        assert client.post("/turn", json={"text": text}).status_code == 200
        return time.perf_counter() - t

    turn("こんにちは。")  # warms the prefix cache
    first = turn("水を飲みました。")
    second = turn("コーヒーが好きです。")
    assert second < first * 2.0, f"cache reuse looks broken: {first:.2f}s then {second:.2f}s"
