"""T2.1 — /ws over the real WebSocket transport.

Same reasoning as `test_api_turn.py`: testing the serializer is not testing the
endpoint. The T1.8 bug that motivated those tests -- a thread-affinity failure
invisible until a real transport was in front of the code -- is exactly the class
of bug a WebSocket pipeline invites, so the round trip is asserted through
Starlette rather than through `OchaSerializer` alone.

`?loopback=1` selects the diagnostic echo pipeline -- no VAD, no models -- so
audio in must come back out byte for byte. The real pipeline is covered by
`test_ws_pipeline.py`, which injects stub ASR and TTS.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest
from fastapi.testclient import TestClient

import ocha.api.main as api_main
from ocha.api.main import app
from ocha.speech.asr import OchaWhisper
from ocha.speech.wire import SAMPLE_RATE, AudioKind, unpack_audio
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
    with client.websocket_connect("/ws?loopback=1") as ws:
        for _ in range(10):
            ws.send_bytes(CHUNK)
        pcm = b""
        sequences: list[int] = []
        exchange_ids = set()
        kinds = set()
        while len(pcm) < len(CHUNK):
            msg = ws.receive()
            if (data := msg.get("bytes")) is not None:
                exchange_id, sequence, kind, chunk = unpack_audio(data)
                exchange_ids.add(exchange_id)
                sequences.append(sequence)
                kinds.add(kind)
                pcm += chunk
    assert len(exchange_ids) == 1
    assert sequences == list(range(len(sequences)))
    assert kinds == {AudioKind.TUTOR}
    assert pcm[: len(CHUNK)] == CHUNK


def test_the_rate_is_pinned_at_16k() -> None:
    """Guards the one constant the iOS client hardcodes on its side."""
    assert SAMPLE_RATE == 16_000


async def test_each_websocket_gets_fresh_asr_lifecycle_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[OchaWhisper] = []
    worker = object()

    class Socket:
        async def accept(self) -> None:
            pass

    class Probe:
        def report(self) -> dict[str, object]:
            return {}

    async def fake_run_session(*args: object, **kwargs: object) -> Probe:
        created.append(cast(OchaWhisper, kwargs["asr"]))
        return Probe()

    for name, value in {
        "worker": worker,
        "conn": object(),
        "scheduler": object(),
        "grammar": object(),
        "llm": StubLlm(),
        "fillers": None,
        "repair_audio": None,
    }.items():
        monkeypatch.setattr(app.state, name, value, raising=False)
    monkeypatch.setattr("ocha.speech.pipeline.run_session", fake_run_session)

    await api_main.ws(Socket())  # type: ignore[arg-type]
    await api_main.ws(Socket())  # type: ignore[arg-type]

    assert created[0] is not created[1]
    assert all(asr._worker is worker for asr in created)
