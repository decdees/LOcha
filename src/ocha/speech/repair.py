"""Visible, pre-synthesised repair for suspicious ASR transcripts."""

from __future__ import annotations

from pipecat.frames.frames import Frame, OutputTransportMessageUrgentFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ocha.speech.asr import AsrRejectedFrame

REPAIR_TEXT = "すみません。もう一度お願いします。"
CHUNK_BYTES = 320


class RepairAudioFrame(TTSAudioRawFrame):
    """Curated retry audio, distinct from filler and tutor audio."""


async def synthesise_repair(tts: object) -> bytes:
    """Render the fixed prompt at startup; never invoke TTS on a rejection path."""
    audio = b""
    async for frame in tts.speak(REPAIR_TEXT):  # type: ignore[attr-defined]
        if isinstance(frame, TTSAudioRawFrame):
            audio += frame.audio
    if not audio:
        raise RuntimeError("VOICEVOX produced no audio for the ASR repair prompt")
    return audio


class AsrRepairProcessor(FrameProcessor):
    def __init__(self, audio: bytes | None, sample_rate: int) -> None:
        super().__init__()
        self._audio = audio
        self._sample_rate = sample_rate

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not isinstance(frame, AsrRejectedFrame):
            await self.push_frame(frame, direction)
            return

        await self.push_frame(
            OutputTransportMessageUrgentFrame(
                message={
                    "type": "asr_rejected",
                    "reason": frame.reason,
                    "text": frame.text,
                    "repair_text": REPAIR_TEXT,
                }
            ),
            FrameDirection.DOWNSTREAM,
        )
        if self._audio is not None:
            for offset in range(0, len(self._audio), CHUNK_BYTES):
                await self.push_frame(
                    RepairAudioFrame(
                        audio=self._audio[offset : offset + CHUNK_BYTES],
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    ),
                    FrameDirection.DOWNSTREAM,
                )
        # Consume the rejection: TutorStage only receives accepted transcripts.
