from __future__ import annotations

from pathlib import Path

from pipecat.frames.frames import (
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language

from ocha.db import connect, migrate
from ocha.db.seed import seed
from ocha.speech.guided_stage import GuidedLessonStage, LessonTargetFrame
from ocha.speech.tts import VoicevoxTTS
from ocha.speech.wire import LessonActionFrame


def _transcript(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "", "2026-08-01T00:00:00Z", language=Language.JA)


class StubVoicevox(VoicevoxTTS):
    def __init__(self) -> None:
        super().__init__(sample_rate=16_000)
        self.spoken: list[str] = []

    def _synthesise(self, text: str) -> bytes:
        self.spoken.append(text)
        return b"\x00\x00" * 160


async def test_initial_guided_target_is_spoken_in_japanese(tmp_path: Path) -> None:
    conn = connect(tmp_path / "audio.db")
    migrate(conn)
    tts = StubVoicevox()
    down, _ = await run_test(Pipeline([GuidedLessonStage(conn), tts]), frames_to_send=[])
    assert tts.spoken == ["こんにちは"]
    assert any(isinstance(frame, TTSAudioRawFrame) for frame in down)


async def test_guided_step_requires_repeat_then_hidden_recall(tmp_path: Path) -> None:
    conn = connect(tmp_path / "guided.db")
    migrate(conn)
    seed(conn)
    reviews_before = conn.execute("SELECT count(*) FROM reviews").fetchone()[0]

    down, _ = await run_test(
        Pipeline([GuidedLessonStage(conn)]),
        frames_to_send=[_transcript("こんにちは"), _transcript("こんにちは。")],
    )
    messages = [
        frame.message
        for frame in down
        if isinstance(frame, OutputTransportMessageUrgentFrame)
        and frame.message.get("type") == "lesson"
    ]

    assert messages[0]["phase"] == "listen"
    assert messages[1]["phase"] == "repeat"
    challenge = next(message for message in messages if message["phase"] == "challenge")
    assert challenge["show_japanese"] is False
    assert challenge["show_romaji"] is False
    assert any(message["phase"] == "success" for message in messages)
    assert (
        conn.execute(
            "SELECT status FROM guided_progress WHERE step_id='greeting-hello'"
        ).fetchone()[0]
        == "completed"
    )
    assert conn.execute("SELECT count(*) FROM reviews").fetchone()[0] == reviews_before
    assert any(isinstance(frame, LessonTargetFrame) for frame in down)


async def test_retry_reveal_and_skip_are_explicit(tmp_path: Path) -> None:
    conn = connect(tmp_path / "actions.db")
    migrate(conn)
    stage = GuidedLessonStage(conn)
    down, _ = await run_test(
        Pipeline([stage]),
        frames_to_send=[
            _transcript("ありがとう"),
            LessonActionFrame(action="reveal", lesson_id="greetings", step_id="greeting-hello"),
            LessonActionFrame(action="skip", lesson_id="greetings", step_id="greeting-hello"),
        ],
    )
    messages = [
        frame.message
        for frame in down
        if isinstance(frame, OutputTransportMessageUrgentFrame)
        and frame.message.get("type") == "lesson"
    ]
    assert any(message["phase"] == "retry" for message in messages)
    revealed = [message for message in messages if message["instruction_en"].startswith("Here")]
    assert revealed and revealed[0]["show_romaji"] is True
    assert (
        conn.execute(
            "SELECT status FROM guided_progress WHERE step_id='greeting-hello'"
        ).fetchone()[0]
        == "skipped"
    )
