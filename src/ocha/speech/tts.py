"""TTS (T2.5) — VOICEVOX Engine over its local HTTP API.

Two calls per sentence: `/audio_query` builds the accent-annotated phrase
structure, then `/synthesis` renders it. The split exists because the query is
*inspectable and editable* -- Phase 3 (§6.1) reads the accent phrases from it as
the reference for pitch scoring, which is the whole reason VOICEVOX was chosen
over a faster neural TTS (ARCHITECTURE §2.2).

**Speaker 13 (青山龍星)** is pinned. It is a Phase 3 dependency, not a voice
preference: VOICEVOX output is the DTW reference for accent scoring, and an octave
gap between reference and learner degrades alignment even after normalisation.
Secondarily, and also not aesthetic -- the learner internalises this voice's
prosody, so it should be a register they can reproduce.

`outputSamplingRate` is set on the query rather than resampling afterwards.
VOICEVOX honours it, so the system stays at one rate end to end and no resampler
sits in the latency path.

## Not a `TTSService`, and that is a measured decision

This began as a subclass of Pipecat's `TTSService`, which aggregates text into
sentences (`TextAggregationMode.SENTENCE` is its default) and would have made T2.4
free. It aggregates correctly -- `match_endofsentence` finds the boundary in
「はい、これをください。」 at the right index -- but it does **not synthesise when it
finds one**. Traced three times: the sentence was pushed at 2.60 s and `run_tts`
was not entered until 3.10 s, twenty milliseconds after the LLM response *ended*.
Neither `reuse_context_id_within_turn=False` nor `push_start_frame=True` changed
it; the deferral is inside the audio-context machinery.

That machinery provides audio contexts, word timestamps and provider metrics, none
of which this needs, and it costs the half second §5.2 rule 2 exists to save. So
this is a plain `FrameProcessor`: a sentence arrives, it is synthesised, the audio
goes out.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections.abc import AsyncIterator
from io import BytesIO

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ocha.speech.wire import SAMPLE_RATE

BASE_URL = "http://127.0.0.1:50021"
SPEAKER = 13  # 青山龍星 — see the module docstring; do not change casually

# 10 ms at 16 kHz. Small chunks keep the client's playback buffer shallow, which is
# what makes barge-in feel immediate rather than "after the current chunk".
CHUNK_FRAMES = 160


class VoicevoxTTS(FrameProcessor):
    """One sentence in, PCM frames out, as soon as the sentence arrives."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        speaker: int = SPEAKER,
        sample_rate: int = SAMPLE_RATE,
        timeout: float = 30.0,
    ) -> None:
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._speaker = speaker
        # Taken in the constructor, not from the StartFrame. The previous version
        # read `TTSService.sample_rate`, which stays 0 until the pipeline starts --
        # so a standalone call asked VOICEVOX for 0 Hz and got an HTTP 500 while
        # every stub-based test stayed green.
        self.sample_rate = sample_rate
        self._timeout = timeout
        self._speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self._speaking = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._speaking:
                self._speaking = False
                await self.push_frame(TTSStoppedFrame())
            await self.push_frame(frame, direction)
            return

        # TranscriptionFrame is a TextFrame too, and synthesising the user's own
        # words back at them is a bug that would sound like a feature.
        if (
            isinstance(frame, TextFrame)
            and not isinstance(frame, TranscriptionFrame | InterimTranscriptionFrame)
            and frame.text.strip()
        ):
            await self.push_frame(frame, direction)  # the client shows the text too
            async for out in self.speak(frame.text):
                await self.push_frame(out)
            return

        await self.push_frame(frame, direction)

    async def speak(self, text: str) -> AsyncIterator[Frame]:
        """Synthesise one piece of text. Public so it can be driven directly."""
        if not self._speaking:
            self._speaking = True
            yield TTSStartedFrame()
        try:
            # urllib in a thread. This is HTTP to another local process, not MLX, so
            # constraint 6 does not apply -- and blocking the loop for the synthesis
            # would stall the very frames this exists to deliver early.
            pcm = await asyncio.to_thread(self._synthesise, text)
        except Exception as exc:  # noqa: BLE001 -- surfaced as a frame, not raised
            yield ErrorFrame(f"VOICEVOX synthesis failed: {exc}")
            return

        step = CHUNK_FRAMES * 2  # 16-bit samples
        for i in range(0, len(pcm), step):
            yield TTSAudioRawFrame(
                audio=pcm[i : i + step], sample_rate=self.sample_rate, num_channels=1
            )

    def _synthesise(self, text: str) -> bytes:
        query = self._post(f"/audio_query?text={urllib.parse.quote(text)}&speaker={self._speaker}")
        params = json.loads(query)
        params["outputSamplingRate"] = self.sample_rate
        params["outputStereo"] = False
        wav = self._post(
            f"/synthesis?speaker={self._speaker}",
            body=json.dumps(params).encode(),
            content_type="application/json",
        )
        # /synthesis returns a RIFF container. The wire format is bare PCM, so the
        # 44-byte header would otherwise be played as audio.
        with wave.open(BytesIO(wav)) as w:
            return bytes(w.readframes(w.getnframes()))

    def _post(self, path: str, body: bytes | None = None, content_type: str | None = None) -> bytes:
        headers = {"Content-Type": content_type} if content_type else {}
        req = urllib.request.Request(
            self._base_url + path, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                return bytes(resp.read())
        except urllib.error.HTTPError as exc:
            # VOICEVOX puts the reason in the body. Without it the caller sees a bare
            # "HTTP Error 500" -- which is how an outputSamplingRate of 0 spent a
            # debugging session looking like a network fault.
            raise RuntimeError(f"{exc.code} from {path}: {exc.read().decode()[:200]}") from exc
