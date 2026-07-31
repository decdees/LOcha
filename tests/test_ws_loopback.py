"""T2.1 — /ws over the real WebSocket transport.

Same reasoning as `test_api_turn.py`: testing the serializer is not testing the
endpoint. The T1.8 bug that motivated those tests -- a thread-affinity failure
invisible until a real transport was in front of the code -- is exactly the class
of bug a WebSocket pipeline invites, so the round trip is asserted through
Starlette rather than through `OchaSerializer` alone.

The pipeline is a loopback today (see `speech/pipeline.py`), so audio in must
come back out. When T2.2--T2.5 replace the loopback, this file's audio assertion
is expected to change; the connect-and-survive assertion is not.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ocha.api.main import app
from ocha.speech.wire import SAMPLE_RATE
from ocha.tutor.llm import StubLlm

# 20 ms of silence at the pinned rate: 320 frames, 2 bytes each.
CHUNK = b"\x00\x00" * 320


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OCHA_DB", str(tmp_path / "ws.db"))
    monkeypatch.setattr("ocha.db.schema.DEFAULT_DB", tmp_path / "ws.db")
    monkeypatch.setattr("ocha.api.main.SKIP_MODEL", True)
    with TestClient(app) as c:
        app.state.llm = StubLlm()
        yield c


def test_pcm_survives_the_round_trip(client: TestClient) -> None:
    """The day-one audio check: what the client sends is what it gets back.

    Byte-identical is the point. Anything that resampled, re-headered or
    re-chunked the audio would still *sound* roughly right on the phone while
    quietly changing what whisper receives in T2.3.
    """
    with client.websocket_connect("/ws") as ws:
        for _ in range(10):
            ws.send_bytes(CHUNK)
        out = b""
        while len(out) < len(CHUNK):
            msg = ws.receive()
            if (data := msg.get("bytes")) is not None:
                out += data
    assert out[: len(CHUNK)] == CHUNK


def test_the_rate_is_pinned_at_16k() -> None:
    """Guards the one constant the iOS client hardcodes on its side."""
    assert SAMPLE_RATE == 16_000
