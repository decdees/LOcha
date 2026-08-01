"""ASR (T2.3) — whisper-large-v3 on MLX, the T0.3 choice.

Written against `mlx_whisper` directly rather than using Pipecat's
`WhisperSTTServiceMLX`, for two reasons:

1. `pipecat.services.whisper.stt` imports `faster_whisper` at module scope even
   for the MLX class, so using it means installing CTranslate2 to run a model we
   already call directly.
2. **Threading.** Standing constraint 6: MLX GPU streams are thread-local to the
   thread that ran `load()`. Transcription therefore runs on the `InferenceWorker`
   thread, which is the thread that loaded both this model and the LLM. It must
   never be called anywhere else -- and no stub-based test would catch it if it
   were, because tests never load MLX.

   Whisper moved onto the worker for the same measured reason the LLM did (T2.6):
   transcription blocks for ~1.0 s, and while it holds the event loop nothing can
   be delivered to the client -- including the filled pause that G1a now depends
   on to cover exactly this gap.

`wants_wav_segments = False`: whisper takes a float32 array, so a WAV header
would be 44 bytes of noise at the front of every utterance.

The model is **not** loaded here. It is warmed at lifespan alongside the LLM
(constraint 3) -- a cold load inside a turn destroys the conversational illusion,
and mlx_whisper caches by repo path, so the first real call is only fast if
something already paid for it.
"""

from __future__ import annotations

import unicodedata
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from ocha.inference import InferenceWorker
from ocha.models import resolve_cached_model
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
# **Those guards are not sufficient.** An end-to-end run produced
# `火が火に火に火に火に...` from a single utterance — T0.3's degenerate repetition,
# on this backend, with the guards active. Open: the transformers mitigation does
# not exist here, so the fix has to be either a post-hoc repetition check on the
# transcript or a different decoding strategy. See benchmarks/voice-loop.md.
CONDITION_ON_PREVIOUS_TEXT = False

# A VAD segment shorter than this is a click, a breath, or a door. Transcribing
# it costs a full whisper pass and yields a hallucinated 「ありがとうございました」,
# which then becomes a conversational turn.
MIN_SEGMENT_S = 0.3

# Whisper hallucinates confident text on near-silence, and its favourite in
# Japanese is 「ご視聴ありがとうございました」 ("thank you for watching") — a YouTube
# sign-off that saturates the training data. Observed twice in an 8-turn
# end-to-end run, each time on a segment that was mostly silence. Transcripts
# matching these are dropped: an invented utterance costs a whole turn and teaches
# the learner that the tutor mishears them.
# A transcript whose text is one short span repeated over and over is the decoder
# looping, not the learner. T0.3 saw it on transformers (557 insertions, 70 s for
# one 4.9 s clip) and fixed it with `no_repeat_ngram_size=4`, which does not exist
# on MLX; an end-to-end run then produced `火が火に火に火に…` here with MLX's own
# guards active. So the check is post-hoc, on the text.
#
# The threshold has to tolerate real Japanese, which repeats short spans
# legitimately -- 「だんだん」, 「いろいろ」, 「はいはい」 -- so it triggers only when a
# 2-4 character span accounts for most of a transcript long enough that the
# repetition cannot be a word.
LOOP_MIN_CHARS = 12
LOOP_MIN_REPEATS = 4
LOOP_COVERAGE = 0.7

HALLUCINATIONS = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録",
)

AsrRejectReason = Literal["known_hallucination", "decoder_loop"]


@dataclass(frozen=True, slots=True)
class AsrDecision:
    text: str
    accepted: bool
    reason: AsrRejectReason | None = None


@dataclass
class AsrRejectedFrame(Frame):
    text: str
    reason: AsrRejectReason


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().rstrip("。！？.!?").strip()


def decide_transcript(text: str) -> AsrDecision:
    """Classify a transcript without silently consuming suspicious speech."""
    cleaned = text.strip()
    normalized = _normalized(cleaned)
    if normalized in {_normalized(value) for value in HALLUCINATIONS}:
        return AsrDecision(cleaned, False, "known_hallucination")
    if is_looping(normalized):
        return AsrDecision(cleaned, False, "decoder_loop")
    return AsrDecision(cleaned, True)


class OchaWhisper(SegmentedSTTService):
    """Transcribes one VAD-delimited segment at a time."""

    def __init__(
        self, *, model: str = MODEL, worker: InferenceWorker | None = None, **kwargs: object
    ) -> None:
        # Settings are declared rather than left NOT_GIVEN: Pipecat logs an ERROR
        # for each unset field, and an error-level line that is always there trains
        # you to ignore error-level lines.
        super().__init__(settings=STTSettings(model=model, language=LANGUAGE), **kwargs)  # type: ignore[arg-type]
        self._model = model
        self._model_path: Path | None = None
        # None only in tests that never load MLX. In the app it is always set, and
        # `_transcribe` running inline is a thread-affinity bug waiting to happen.
        self._worker = worker

    @property
    def wants_wav_segments(self) -> bool:
        return False  # raw 16-bit PCM; whisper wants samples, not a container

    def warm(self) -> None:
        """Load and compile by transcribing 100 ms of silence.

        **Must be called on the worker thread** -- pass it to `InferenceWorker.start`
        as a loader. Silence is deliberate: the point is to pay the load and
        graph-compile cost, not to get a result.
        """
        import mlx_whisper

        self._model_path = resolve_cached_model(self._model)
        mlx_whisper.transcribe(
            np.zeros(int(0.1 * (self.sample_rate or SAMPLE_RATE)), dtype=np.float32),
            path_or_hf_repo=self._model_path,
            language=LANGUAGE,
        )

    # See the note on VoicevoxTTS.run_tts: the base declares this as a coroutine
    # returning a generator, and every implementation is an async generator.
    async def run_stt(  # type: ignore[override]
        self, audio: bytes
    ) -> AsyncGenerator[Frame | None, None]:
        samples = np.frombuffer(audio, dtype=np.int16)
        # `or SAMPLE_RATE`: self.sample_rate is 0 until the pipeline's StartFrame
        # reaches start(). In the pipeline it is always set; standalone it is not,
        # and a threshold of zero would let every click through.
        if len(samples) < MIN_SEGMENT_S * (self.sample_rate or SAMPLE_RATE):
            yield None
            return

        # int16 -> float32 in [-1, 1). Dividing by 32768 rather than 32767 keeps
        # the mapping exact in binary and matches what whisper's own loader does.
        decision = decide_transcript(await self._transcribe(samples))
        if not decision.accepted:
            assert decision.reason is not None
            yield AsrRejectedFrame(text=decision.text, reason=decision.reason)
            return
        if not decision.text:
            yield None
            return
        yield TranscriptionFrame(decision.text, "", time_now_iso8601(), language=LANGUAGE)

    async def _transcribe(self, samples: np.ndarray) -> str:
        def work() -> str:
            import mlx_whisper

            if self._model_path is None:
                self._model_path = resolve_cached_model(self._model)

            result = mlx_whisper.transcribe(
                samples.astype(np.float32) / 32768.0,
                path_or_hf_repo=self._model_path,
                language=LANGUAGE,
                condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            )
            return str(result["text"]).strip()

        if self._worker is None:
            raise RuntimeError(
                "OchaWhisper has no InferenceWorker. MLX streams are thread-local "
                "(constraint 6); running inline would work until it did not."
            )
        return str(await self._worker.call(work))


def is_looping(text: str) -> bool:
    """True when the transcript is one short span repeated, i.e. a decoder loop.

    Deliberately conservative: a false positive silently drops a real utterance,
    which costs a turn, while a false negative produces a visibly absurd reply the
    learner will ignore. Better to let one through than to eat someone's sentence.
    """
    stripped = text.strip()
    if len(stripped) < LOOP_MIN_CHARS:
        return False
    for span in range(2, 5):
        for start in range(span):
            unit = stripped[start : start + span]
            if not unit.strip():
                continue
            repeats = stripped.count(unit)
            if repeats >= LOOP_MIN_REPEATS and repeats * span / len(stripped) >= LOOP_COVERAGE:
                return True
    return False
