"""Deterministic listen, repeat, and recall flow for absolute beginners."""

from __future__ import annotations

import sqlite3
from typing import Literal

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ocha.lessons import (
    current_step,
    load_lessons,
    record_progress,
    transcript_matches,
)
from ocha.speech.wire import LessonActionFrame


class LessonTargetFrame(TextFrame):
    """Japanese text intended for TTS but not the conversation reply renderer."""


Phase = Literal["repeat", "challenge"]


class GuidedLessonStage(FrameProcessor):
    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__()
        self._conn = conn
        self._lessons = load_lessons()
        self._step = current_step(conn, self._lessons)
        self._phase: Phase = "repeat"

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._present()
            return
        if isinstance(frame, LessonActionFrame):
            await self._action(frame)
            return
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            await self.push_frame(frame, direction)
            await self._attempt(frame.text)
            return
        await self.push_frame(frame, direction)

    async def _message(
        self,
        phase: str,
        instruction: str,
        *,
        show_japanese: bool = True,
        show_romaji: bool = True,
    ) -> None:
        step = self._step
        if step is None:
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={
                        "type": "lesson",
                        "lesson_id": "course",
                        "step_id": "complete",
                        "phase": "complete",
                        "instruction_en": "You completed the beginner course.",
                        "japanese": "",
                        "romaji": "",
                        "meaning_en": "",
                        "show_japanese": False,
                        "show_romaji": False,
                    }
                )
            )
            return
        await self.push_frame(
            OutputTransportMessageUrgentFrame(
                message={
                    "type": "lesson",
                    "lesson_id": step.module_id,
                    "lesson_title": step.module_title,
                    "step_id": step.id,
                    "phase": phase,
                    "instruction_en": instruction,
                    "japanese": step.japanese,
                    "romaji": step.romaji,
                    "meaning_en": step.meaning_en,
                    "show_japanese": show_japanese,
                    "show_romaji": show_romaji,
                }
            )
        )

    async def _present(self) -> None:
        if self._step is None:
            await self._message("complete", "You completed the beginner course.")
            return
        self._phase = "repeat"
        await self._message("listen", "Listen.")
        await self.push_frame(LessonTargetFrame(self._step.japanese))
        # VoicevoxTTS keeps a speaking context open until this boundary. Without
        # it the transport never announces that lesson audio ended, so the PWA
        # cannot safely release its buffered PCM before inviting the learner to
        # speak.
        await self.push_frame(LLMFullResponseEndFrame())
        await self._message("repeat", "Now repeat. Tap the microphone below, then speak.")

    async def _attempt(self, transcript: str) -> None:
        step = self._step
        if step is None:
            return
        if not transcript_matches(step, transcript):
            await self._message(
                "retry", "I heard something different. Tap the microphone and try again."
            )
            return
        if self._phase == "repeat":
            self._phase = "challenge"
            await self._message(
                "challenge",
                "Now say it from the English meaning. Tap the microphone when you are ready.",
                show_japanese=False,
                show_romaji=False,
            )
            return
        record_progress(self._conn, step.id, "completed")
        await self._message("success", "Good job.")
        self._step = current_step(self._conn, self._lessons)
        await self._present()

    async def _action(self, frame: LessonActionFrame) -> None:
        step = self._step
        if step is None:
            return
        if frame.lesson_id != step.module_id or frame.step_id != step.id:
            return
        if frame.action == "replay":
            await self._present()
        elif frame.action == "reveal":
            await self._message("challenge", "Here is the answer. Try it again.")
        else:
            record_progress(self._conn, step.id, "skipped")
            await self._message("success", "Skipped.")
            self._step = current_step(self._conn, self._lessons)
            await self._present()
