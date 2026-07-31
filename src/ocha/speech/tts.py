"""TTS (T2.5) — VOICEVOX Engine over its local HTTP API.

Pipecat has no VOICEVOX service, so this is a custom one. Two calls per
sentence: `/audio_query` builds the accent-annotated phrase structure, then
`/synthesis` renders it. The split exists because the query is *inspectable and
editable* -- Phase 3 (§6.1) reads the accent phrases from it as the reference for
pitch scoring, which is the whole reason VOICEVOX was chosen over a faster
neural TTS (ARCHITECTURE §2.2).

**Speaker 13 (青山龍星)** is pinned. It is a Phase 3 dependency, not a voice
preference: VOICEVOX output is the DTW reference for accent scoring, and an
octave gap between reference and learner degrades alignment even after
normalisation. Secondary and also not aesthetic -- the learner internalises this
voice's prosody, so it should be a register they can reproduce.

`outputSamplingRate` is set on the query rather than resampling afterwards.
VOICEVOX honours it, so the whole system stays at one rate end to end and no
resampler sits in the latency path.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections.abc import AsyncGenerator
from io import BytesIO

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from ocha.speech.wire import SAMPLE_RATE

BASE_URL = "http://127.0.0.1:50021"
SPEAKER = 13  # 青山龍星 — see the module docstring; do not change casually

# 10 ms at 16 kHz. Small chunks keep the client's playback buffer shallow, which
# is what makes barge-in feel immediate rather than "after the current chunk".
CHUNK_FRAMES = 160


class VoicevoxTTS(TTSService):
    """One sentence in, PCM frames out."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        speaker: int = SPEAKER,
        timeout: float = 30.0,
        **kwargs: object,
    ) -> None:
        # aggregate_sentences is the default: Pipecat splits on 。！？ for us, which
        # is ARCHITECTURE §5.2 rule 2 -- synthesise 「そうですね。」 while the model
        # is still generating the rest. T2.4 is therefore configuration, not code.
        super().__init__(
            settings=TTSSettings(model="voicevox", voice=str(speaker), language=Language.JA),
            **kwargs,  # type: ignore[arg-type]
        )
        self._base_url = base_url.rstrip("/")
        self._speaker = speaker
        self._timeout = timeout

    def can_generate_metrics(self) -> bool:
        return True

    # The override ignore is upstream's shape, not ours: TTSService declares
    # `async def run_tts(...) -> AsyncGenerator[...]` with a `pass` body, which types
    # as a coroutine *returning* a generator. Every real implementation, Pipecat's
    # own included, is an async generator instead.
    async def run_tts(  # type: ignore[override]
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame | None, None]:
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()
        try:
            # urllib in a thread, not on the event loop. This is HTTP to another
            # local process, not MLX -- constraint 6 does not apply, and blocking
            # the loop for the synthesis duration would stall the probe's own
            # state messages, which are what G1a depends on.
            pcm = await asyncio.to_thread(self._synthesise, text)
        except Exception as exc:  # noqa: BLE001 -- surfaced as a frame, not raised
            yield ErrorFrame(f"VOICEVOX synthesis failed: {exc}")
            yield TTSStoppedFrame()
            return

        await self.stop_ttfb_metrics()
        step = CHUNK_FRAMES * 2  # 16-bit samples
        for i in range(0, len(pcm), step):
            yield TTSAudioRawFrame(
                audio=pcm[i : i + step],
                sample_rate=self.sample_rate or SAMPLE_RATE,
                num_channels=1,
            )
        yield TTSStoppedFrame()

    def _synthesise(self, text: str) -> bytes:
        # self.sample_rate is 0 until the pipeline's StartFrame reaches start(), and
        # a query asking VOICEVOX for 0 Hz returns HTTP 500. In the pipeline it is
        # always set (and always equals the wire rate); the fallback keeps the
        # service usable standalone, which is how it gets diagnosed.
        rate = self.sample_rate or SAMPLE_RATE
        query = self._post(f"/audio_query?text={urllib.parse.quote(text)}&speaker={self._speaker}")
        params = json.loads(query)
        params["outputSamplingRate"] = rate
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
            # VOICEVOX puts the reason in the body. Without it the caller sees a
            # bare "HTTP Error 500" -- which is how an outputSamplingRate of 0 spent
            # a debugging session looking like a network fault.
            raise RuntimeError(f"{exc.code} from {path}: {exc.read().decode()[:200]}") from exc
