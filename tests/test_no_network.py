"""T1.10 — NFR-3: no outbound network in any request path."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ocha.api.main import app
from ocha.net_guard import OutboundNetworkError, is_allowed, no_outbound_network
from ocha.tutor.llm import StubLlm


def test_loopback_and_tailscale_are_allowed() -> None:
    assert is_allowed("127.0.0.1")  # local model server, VOICEVOX
    assert is_allowed("localhost")
    assert is_allowed("100.101.102.103")  # Tailscale CGNAT
    assert is_allowed("::1")


def test_public_addresses_and_hostnames_are_not() -> None:
    assert not is_allowed("142.250.185.078")  # a public IP
    assert not is_allowed("api.openai.com")  # a hostname means DNS we don't control
    assert not is_allowed("8.8.8.8")
    assert not is_allowed("192.168.1.5")  # LAN is not Tailscale


def test_the_guard_actually_catches_an_outbound_call() -> None:
    """The guard must itself be proven. TASKS.md T1.10: deliberately adding an
    outbound call has to make this fail."""
    with no_outbound_network(), pytest.raises(OutboundNetworkError):
        socket.create_connection(("142.250.185.78", 443), timeout=1)


def test_the_guard_is_removed_afterwards() -> None:
    """A leaked monkeypatch would silently disable every later test."""
    with no_outbound_network():
        pass
    assert socket.socket.connect.__qualname__ != "no_outbound_network.<locals>.guard"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OCHA_DB", str(tmp_path / "n.db"))
    monkeypatch.setattr("ocha.db.schema.DEFAULT_DB", tmp_path / "n.db")
    monkeypatch.setattr("ocha.api.main.SKIP_MODEL", True)
    with TestClient(app) as c:
        app.state.llm = StubLlm()
        yield c


def test_a_turn_makes_no_outbound_connection(client: TestClient) -> None:
    """The real assertion: a full turn through the app touches nothing outbound."""
    with no_outbound_network():
        r = client.post("/turn", json={"text": "こんにちは。"})
    assert r.status_code == 200


def test_health_makes_no_outbound_connection(client: TestClient) -> None:
    with no_outbound_network():
        assert client.get("/health").status_code == 200


def test_grammar_turn_makes_no_outbound_connection(client: TestClient) -> None:
    """The firewall must never reach out for an explanation it does not have."""
    app.state.llm = StubLlm(reply="[GRAMMAR_QUERY]")
    with no_outbound_network():
        r = client.post("/turn", json={"text": "what is the causative passive"})
    assert r.status_code == 200
    assert r.json()["grammar"]["kind"] == "not_documented"
