"""T2.2–T2.5 — the assembled pipeline, with only the two model stages stubbed.

Real here: **silero VAD driven by an actual recording**, the `TutorStage` (and
through it `build_context`, the firewall, usage detection and FSRS scoring), the
probe, and the frame plumbing between all of them. Stubbed: whisper and VOICEVOX,
because a 3 GB model and an external HTTP service cannot run on every commit.

The audio is `benchmarks/corpus/03.wav` — the T0.2 corpus, 16 kHz mono, already
the wire's rate. Synthetic noise was the alternative and is worse: silero may or
may not fire on it, so a green test would say nothing about whether VAD is wired
up at all.

Driven through `run_test` rather than a WebSocket. `TestClient.websocket_connect`
gives a `receive()` with no timeout, so any wiring mistake presents as a hung
suite instead of a failure — which is exactly what happened while writing this.
The transport itself is covered by `test_ws_loopback.py`.

This targets one class of bug specifically: **the frames stages exchange**. T1.8
taught that testing the function is not testing the endpoint; a Pipecat pipeline
adds another layer, where a stage can be individually correct and still emit a
frame type the next stage ignores. Two live examples, both found by this file:
`VADProcessor` emits `VADUserStartedSpeakingFrame`, which is *not* a subclass of
`UserStartedSpeakingFrame`; and `TTSService` consumes `LLMTextFrame`, so the probe
sitting after it saw no LLM text at all.
"""

from __future__ import annotations

import sqlite3
import wave
from collections.abc import AsyncGenerator, Iterator, Sequence
from pathlib import Path

import pytest
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from ocha.db import connect, migrate
from ocha.db.seed import seed
from ocha.scheduling import ItemScheduler
from ocha.speech.asr import AsrRejectedFrame, decide_transcript
from ocha.speech.probe import TurnStateProbe
from ocha.speech.repair import REPAIR_TEXT, AsrRepairProcessor, RepairAudioFrame
from ocha.speech.tts import VoicevoxTTS
from ocha.speech.tutor_stage import TutorStage
from ocha.speech.wire import SAMPLE_RATE
from ocha.tutor.firewall import SENTINEL
from ocha.tutor.grammar import load_grammar
from ocha.tutor.llm import LlmService, StubLlm

CORPUS = Path(__file__).resolve().parent.parent / "benchmarks" / "corpus" / "03.wav"
CHUNK = 320 * 2  # 20 ms of 16-bit mono at 16 kHz
TRAILING_SILENCE_CHUNKS = 60  # 1.2 s — comfortably past VADParams.stop_secs


class StubWhisper(SegmentedSTTService):
    """A fixed transcript for whatever segment VAD hands over.

    Still a `SegmentedSTTService`, deliberately: the thing under test is that VAD
    segmentation reaches an STT service at all, which a plain FrameProcessor
    returning a TranscriptionFrame would not exercise.
    """

    def __init__(self, transcript: str, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.transcript = transcript
        self.segments: list[int] = []

    @property
    def wants_wav_segments(self) -> bool:
        return False

    async def run_stt(  # type: ignore[override]
        self, audio: bytes
    ) -> AsyncGenerator[Frame | None, None]:
        self.segments.append(len(audio))
        # The real service's hallucination guard, applied here too -- a stub that
        # skips it would let the guard rot without any test noticing.
        decision = decide_transcript(self.transcript)
        if not decision.accepted:
            assert decision.reason is not None
            yield AsrRejectedFrame(text=decision.text, reason=decision.reason)
            return
        yield TranscriptionFrame(self.transcript, "", time_now_iso8601(), language=Language.JA)


class StubVoicevox(VoicevoxTTS):
    """The real service with only the HTTP call replaced.

    A subclass rather than a separate fake, so these tests exercise the real frame
    handling -- which sentences get synthesised, which do not, and when the started
    and stopped frames are emitted. A hand-written fake would have agreed with
    whatever the pipeline did.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.spoken: list[str] = []

    def _synthesise(self, text: str) -> bytes:
        self.spoken.append(text)
        return b"\x00\x00" * 160  # 10 ms of silence


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "pipeline.db")
    migrate(conn)
    seed(conn)
    yield conn
    conn.close()


class Rig:
    """The pipeline plus handles on the stubs, so tests can assert on both ends."""

    def __init__(self, conn: sqlite3.Connection, llm: LlmService) -> None:
        self.asr = StubWhisper("今日はご飯を食べました。", sample_rate=SAMPLE_RATE)
        self.tts = StubVoicevox(sample_rate=SAMPLE_RATE)
        self.probe = TurnStateProbe(emit_state=True)
        self.pipeline = Pipeline(
            [
                VADProcessor(vad_analyzer=SileroVADAnalyzer()),
                self.asr,
                AsrRepairProcessor(b"\x00\x00" * 160, SAMPLE_RATE),
                TutorStage(conn, ItemScheduler(conn), load_grammar(), llm),
                self.tts,
                self.probe,
            ]
        )
        self.down: Sequence[Frame] = []

    async def speak(self) -> None:
        """Play the recording in, then enough silence for VAD to end the turn."""
        with wave.open(str(CORPUS)) as w:
            assert w.getframerate() == SAMPLE_RATE, "corpus must match the wire rate"
            audio = w.readframes(w.getnframes())
        frames: list[Frame] = [
            InputAudioRawFrame(audio=audio[i : i + CHUNK], sample_rate=SAMPLE_RATE, num_channels=1)
            for i in range(0, len(audio), CHUNK)
        ]
        frames += [
            InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=SAMPLE_RATE, num_channels=1)
            for _ in range(TRAILING_SILENCE_CHUNKS)
        ]
        self.down, _ = await run_test(self.pipeline, frames_to_send=frames)

    def messages(self) -> list[dict[str, object]]:
        return [f.message for f in self.down if isinstance(f, OutputTransportMessageUrgentFrame)]


async def test_a_spoken_utterance_produces_a_spoken_reply(db: sqlite3.Connection) -> None:
    """The whole chain: VAD segments, ASR fires, the tutor replies, TTS speaks."""
    rig = Rig(db, StubLlm())
    await rig.speak()

    assert rig.asr.segments, "VAD never handed a segment to the ASR service"
    assert rig.tts.spoken, "the tutor's reply never reached TTS"
    assert any(isinstance(f, TTSAudioRawFrame) for f in rig.down), "no audio came out"


async def test_the_reply_is_synthesised_sentence_by_sentence(db: sqlite3.Connection) -> None:
    """§5.2 rule 2. Synthesis must start on the first sentence, not the whole reply.

    `StubLlm`'s default reply is two sentences, so two synthesis calls is the
    evidence -- one call would mean the reply was buffered whole, which is the
    named worst failure mode in the voice loop.
    """
    rig = Rig(db, StubLlm())
    await rig.speak()
    assert len(rig.tts.spoken) >= 2, f"synthesised as one block: {rig.tts.spoken}"


async def test_the_probe_can_actually_measure_g1b(db: sqlite3.Connection) -> None:
    """The instrument must work on the assembled pipeline, not just in isolation.

    Both of these were None before this test existed: `VADProcessor` emits VAD
    frames the probe did not map, and `TTSService` consumes the LLM text frames it
    was watching for. A probe that reports None is indistinguishable from a
    pipeline that never ran.
    """
    rig = Rig(db, StubLlm())
    await rig.speak()

    report = rig.probe.report()
    assert report["voice_to_first_audio_s"] is not None, "G1b is unmeasurable"
    assert report["llm_ttft_s"] is not None
    for name in ("asr_s", "llm_ttft_s", "tts_s", "voice_to_first_audio_s"):
        value = report[name]
        assert isinstance(value, float) and value >= 0, f"{name} is not a duration: {value}"


async def test_the_client_is_told_what_is_happening(db: sqlite3.Connection) -> None:
    """G1a: the states must be sent, not merely recorded."""
    rig = Rig(db, StubLlm())
    await rig.speak()

    states = [m["state"] for m in rig.messages() if m.get("type") == "state"]
    assert "listening" in states, "VAD frames are not reaching the probe"
    assert "thinking" in states
    assert "speaking" in states


async def test_the_firewall_holds_over_the_voice_path(db: sqlite3.Connection) -> None:
    """The critical one. A grammar query yields curated text and NO speech.

    The voice path is a second route by which model output could reach the user,
    so FR-5 has to be asserted here and not only over HTTP -- a firewall that
    holds for `POST /turn` and leaks through the pipeline is not a firewall. The
    stub emits the sentinel plus an explanation of its own; neither that text nor
    any synthesis of it may appear.
    """
    leak = "は marks the topic, actually"
    rig = Rig(db, StubLlm(reply=f"{SENTINEL} {leak}"))
    await rig.speak()

    grammar = [m for m in rig.messages() if m.get("type") == "grammar"]
    assert grammar, "no grammar payload was delivered"
    assert leak not in repr(grammar), "model text reached the client"
    assert not rig.tts.spoken, f"a grammar answer was spoken aloud: {rig.tts.spoken}"
    assert not any(isinstance(f, TTSAudioRawFrame) for f in rig.down), "audio for a grammar answer"


@pytest.mark.parametrize(
    "reply",
    [
        f"これは普通の返事です。{SENTINEL} は話題を示します。",
        f"一文目です。二文目です。{SENTINEL}",
        "一文目です。GRAMMAR_QUERY は話題を示します。",
    ],
)
async def test_a_late_sentinel_quarantines_the_complete_voice_reply(
    db: sqlite3.Connection, reply: str
) -> None:
    rig = Rig(db, StubLlm(reply=reply))
    await rig.speak()

    assert not rig.tts.spoken, f"model text escaped before the firewall: {rig.tts.spoken}"
    assert not any(isinstance(f, TTSAudioRawFrame) for f in rig.down)


async def test_a_hallucinated_transcript_requests_a_visible_retry(
    db: sqlite3.Connection,
) -> None:
    """Whisper invents 「ご視聴ありがとうございました」 on near-silence.

    Observed twice in an 8-turn end-to-end run. An invented utterance is worse
    than no utterance: the tutor answers something the learner never said, which
    reads as the tutor mishearing them, and it consumes a turn's FSRS scoring.
    """
    rig = Rig(db, StubLlm())
    rig.asr.transcript = "ご視聴ありがとうございました"
    await rig.speak()
    assert not rig.tts.spoken, "the tutor replied to a hallucination"
    rejected = [m for m in rig.messages() if m.get("type") == "asr_rejected"]
    assert rejected == [
        {
            "type": "asr_rejected",
            "reason": "known_hallucination",
            "text": "ご視聴ありがとうございました",
            "repair_text": REPAIR_TEXT,
        }
    ]
    assert any(isinstance(frame, RepairAudioFrame) for frame in rig.down)
    assert db.execute("SELECT count(*) FROM turns").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM item_observations").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM reviews").fetchone()[0] == 0
