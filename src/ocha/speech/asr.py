"""ASR (T2.3) — whisper-large-v3 on MLX, the T0.3 choice.

Written against `mlx_whisper` directly rather than using Pipecat's
`WhisperSTTServiceMLX`, for two reasons:

1. `pipecat.services.whisper.stt` imports `faster_whisper` at module scope even
   for the MLX class, so using it means installing CTranslate2 to run a model we
   already call directly.
2. **Threading.** Standing constraint 6: MLX GPU streams are thread-local to the
   thread that ran `load()`. `run_stt` is awaited on the event loop, which is
   where the LLM is loaded during lifespan, so both models stay on one thread.
   Anything that moved transcription to a threadpool would fail at runtime with
   `There is no Stream(gpu, 1) in current thread` -- and no stub-based test would
   catch it, because tests never load MLX.

`wants_wav_segments = False`: whisper takes a float32 array, so a WAV header
would be 44 bytes of noise at the front of every utterance.

The model is **not** loaded here. It is warmed at lifespan alongside the LLM
(constraint 3) -- a cold load inside a turn destroys the conversational illusion,
and mlx_whisper caches by repo path, so the first real call is only fast if
something already paid for it.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import numpy as np
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from ocha.speech.wire import SAMPLE_RATE

MODEL = "mlx-community/whisper-large-v3-mlx"
LANGUAGE = Language.JA

# T0.3 recorded `no_repeat_ngram_size=4` as load-bearing: without it large-v3
# degenerated on some utterances into hundreds of repetitions (557 insertions, 70 s
# for one 4.9 s clip). **That parameter does not exist on this runtime.** It is a
# `transformers` generation flag, and T0.3 measured the kotoba candidates on
# transformers while large-v3 ran on MLX -- `mlx_whisper.transcribe` raises
# `DecodingOptions.__init__() got an unexpected keyword argument`. The benchmark
# script that produced the shipped numbers (benchmarks/contention.py) passed no
# such argument, so the measured 1.25 s and 2.56% CER are from a run without it.
#
# MLX's equivalent guards are these, and they are on by default:
#   compression_ratio_threshold  -- a segment whose gzip ratio is too high is a
#                                   repetition loop; decoding is retried
#   temperature fallback         -- retries at rising temperature on failure
#   condition_on_previous_text   -- disabled below, because carrying context
#                                   across turns is what lets a loop persist
# No looping was observed across the 8-turn T0.7 and T0.9 runs on this path.
CONDITION_ON_PREVIOUS_TEXT = False

# A VAD segment shorter than this is a click, a breath, or a door. Transcribing
# it costs a full whisper pass and yields a hallucinated 「ありがとうございました」,
# which then becomes a conversational turn.
MIN_SEGMENT_S = 0.3


class OchaWhisper(SegmentedSTTService):
    """Transcribes one VAD-delimited segment at a time."""

    def __init__(self, *, model: str = MODEL, **kwargs: object) -> None:
        # Settings are declared rather than left NOT_GIVEN: Pipecat logs an ERROR
        # for each unset field, and an error-level line that is always there trains
        # you to ignore error-level lines.
        super().__init__(settings=STTSettings(model=model, language=LANGUAGE), **kwargs)  # type: ignore[arg-type]
        self._model = model

    @property
    def wants_wav_segments(self) -> bool:
        return False  # raw 16-bit PCM; whisper wants samples, not a container

    def warm(self) -> None:
        """Load and compile by transcribing 100 ms of silence.

        Called from lifespan on the event-loop thread. Silence is deliberate:
        the point is to pay the load and graph-compile cost, not to get a result.
        """
        import mlx_whisper

        mlx_whisper.transcribe(
            np.zeros(int(0.1 * (self.sample_rate or SAMPLE_RATE)), dtype=np.float32),
            path_or_hf_repo=self._model,
            language=LANGUAGE,
        )

    # See the note on VoicevoxTTS.run_tts: the base declares this as a coroutine
    # returning a generator, and every implementation is an async generator.
    async def run_stt(  # type: ignore[override]
        self, audio: bytes
    ) -> AsyncGenerator[Frame | None, None]:
        import mlx_whisper

        samples = np.frombuffer(audio, dtype=np.int16)
        # `or SAMPLE_RATE`: self.sample_rate is 0 until the pipeline's StartFrame
        # reaches start(). In the pipeline it is always set; standalone it is not,
        # and a threshold of zero would let every click through.
        if len(samples) < MIN_SEGMENT_S * (self.sample_rate or SAMPLE_RATE):
            yield None
            return

        # int16 -> float32 in [-1, 1). Dividing by 32768 rather than 32767 keeps
        # the mapping exact in binary and matches what whisper's own loader does.
        result = mlx_whisper.transcribe(
            samples.astype(np.float32) / 32768.0,
            path_or_hf_repo=self._model,
            language=LANGUAGE,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        )
        text = str(result["text"]).strip()
        if not text:
            yield None
            return
        yield TranscriptionFrame(text, "", time_now_iso8601(), language=LANGUAGE)
