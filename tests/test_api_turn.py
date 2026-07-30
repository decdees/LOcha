"""POST /turn over the real HTTP transport.

These exist because the T1.8 unit tests called run_turn() directly and therefore
never crossed the transport boundary. That hid a genuine production bug: FastAPI
dispatches sync endpoints across a threadpool, and the sqlite3 connection opened
during lifespan startup is thread-bound, so every /turn returned 500 with
"SQLite objects created in a thread can only be used in that same thread".
Testing the function is not testing the endpoint.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ocha.api.main import app
from ocha.tutor.firewall import SENTINEL
from ocha.tutor.llm import StubLlm


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OCHA_DB", str(tmp_path / "api.db"))
    monkeypatch.setattr("ocha.db.schema.DEFAULT_DB", tmp_path / "api.db")
    monkeypatch.setattr("ocha.api.main.SKIP_MODEL", True)
    with TestClient(app) as c:
        app.state.llm = StubLlm()  # the real model is 14.2 GB
        yield c


def test_turn_returns_200_over_http(client: TestClient) -> None:
    """The regression test for the threadpool bug."""
    r = client.post("/turn", json={"text": "こんにちは。"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] > 0
    assert body["turn_id"] > 0
    assert body["reply"]


def test_session_is_reused_across_turns(client: TestClient) -> None:
    first = client.post("/turn", json={"text": "こんにちは。"}).json()
    second = client.post(
        "/turn", json={"text": "水を飲みました。", "session_id": first["session_id"]}
    ).json()
    assert second["session_id"] == first["session_id"]
    assert second["turn_id"] > first["turn_id"]


def test_unknown_session_is_404_not_500(client: TestClient) -> None:
    r = client.post("/turn", json={"text": "こんにちは。", "session_id": 999999})
    assert r.status_code == 404


def test_empty_text_is_rejected(client: TestClient) -> None:
    assert client.post("/turn", json={"text": ""}).status_code == 422


def test_grammar_question_serves_reference_not_model_text(client: TestClient) -> None:
    """FR-5 at the HTTP boundary -- the payload the user actually receives."""
    app.state.llm = StubLlm(reply=f"{SENTINEL} は marks the topic and が the subject.")
    r = client.post("/turn", json={"text": "What is the difference between は and が?"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] is None
    assert body["grammar"] is not None
    assert body["grammar"]["entry_id"] == "particle_wa_ga"
    # nothing the model wrote may appear anywhere in the serialised response
    assert SENTINEL not in r.text
    assert "the subject" not in r.text or "marks the subject" in body["grammar"]["text"]


def test_grammar_payload_has_no_field_for_model_text(client: TestClient) -> None:
    app.state.llm = StubLlm(reply=SENTINEL)
    body = client.post("/turn", json={"text": "は vs が?"}).json()
    assert set(body["grammar"]) == {
        "kind",
        "text",
        "entry_id",
        "examples",
        "hindi_contrast",
        "interference_warning",
    }


def test_ten_turns_over_http(client: TestClient) -> None:
    """The T1.8 acceptance criterion, driven the way a client would."""
    session = None
    for n in range(10):
        payload: dict[str, object] = {"text": f"これは{n}です。"}
        if session is not None:
            payload["session_id"] = session
        r = client.post("/turn", json=payload)
        assert r.status_code == 200, r.text
        session = r.json()["session_id"]

    health = client.get("/health").json()
    assert health["grammar_entries"] == 20
