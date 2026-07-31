"""T2.5 — the real VOICEVOX HTTP contract. Slow: needs the engine on :50021.

The stub-based pipeline tests prove the frames connect. They cannot prove the
service works, and they didn't: `self.sample_rate` is 0 until the pipeline's
StartFrame reaches `start()`, so a standalone call asked VOICEVOX to synthesise at
0 Hz and got an HTTP 500. Every stubbed test was green.

That is the same lesson as T1.8's threadpool bug and this suite's `-m slow`
marker existing at all -- a stub cannot fail the way the real thing fails.
"""

from __future__ import annotations

import pytest
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame

from ocha.speech.tts import VoicevoxTTS
from ocha.speech.wire import SAMPLE_RATE

pytestmark = pytest.mark.slow


async def test_synthesis_returns_bare_pcm_at_the_wire_rate() -> None:
    tts = VoicevoxTTS(sample_rate=SAMPLE_RATE)
    audio = b""
    errors: list[str] = []
    async for frame in tts.speak("そうですね。"):
        if isinstance(frame, TTSAudioRawFrame):
            assert frame.sample_rate == SAMPLE_RATE, "VOICEVOX ignored outputSamplingRate"
            assert frame.num_channels == 1
            audio += frame.audio
        elif isinstance(frame, ErrorFrame):
            errors.append(frame.error)

    assert not errors, errors
    # 「そうですね。」 is 6 morae; anything under half a second means the audio was
    # truncated, and anything over three seconds means it is not this sentence.
    seconds = len(audio) / 2 / SAMPLE_RATE
    assert 0.5 < seconds < 3.0, f"implausible duration: {seconds:.2f}s"
    # No RIFF header: the client plays these bytes straight into a buffer.
    assert not audio.startswith(b"RIFF")


async def test_a_bad_request_surfaces_the_engine_s_reason() -> None:
    """A bare "HTTP Error 500" is what made the 0 Hz bug look like a network fault."""
    tts = VoicevoxTTS(sample_rate=SAMPLE_RATE, speaker=99999)
    errors = [f.error async for f in tts.speak("テスト。") if isinstance(f, ErrorFrame)]
    assert errors, "an unknown speaker should have failed"
    assert any(len(e) > len("VOICEVOX synthesis failed: HTTP Error 500") for e in errors), errors
