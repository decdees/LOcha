"""The first real voice turn (T2.6). Slow: loads whisper AND the LLM, needs VOICEVOX.

Everything is real here -- `whisper-large-v3` on MLX, `Qwen3.5-9B-4bit`, VOICEVOX
speaker 13, silero VAD, and a recording of the actual learner's voice. This is
the test the stub suite cannot be: T0.3 measured ASR alone, T0.4 measured the LLM
alone, and T0.7 measured them co-resident *through a script*. None of them ran
the pipeline.

It also re-measures §5.1 from inside the pipeline rather than by summing stage
benchmarks, which is what PRD G1b asks for. The number printed here is directly
comparable to T0.7's 2.53 s.

Not asserting a specific transcript. The corpus is accented beginner Japanese and
T0.3 already established a 2.56% CER on it; pinning the exact string here would
turn an ASR accuracy question into a brittle equality check. What is asserted is
that a transcript happened, that it is Japanese, and that audio came back.
"""

from __future__ import annotations

import sqlite3
import wave
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.tests.utils import run_test

from ocha.db import connect, migrate
from ocha.db.seed import seed
from ocha.inference import InferenceWorker, WorkerLlm
from ocha.scheduling import ItemScheduler
from ocha.speech.asr import OchaWhisper
from ocha.speech.filler import FillerAudioFrame, FillerBank, FillerProcessor, FillerState
from ocha.speech.pipeline import VAD_PARAMS
from ocha.speech.probe import Spans, TurnStateProbe
from ocha.speech.tts import VoicevoxTTS
from ocha.speech.tutor_stage import TutorStage
from ocha.speech.wire import SAMPLE_RATE
from ocha.turnstate import TurnTimeline
from ocha.tutor.grammar import load_grammar
from ocha.tutor.llm import MlxLlm

pytestmark = pytest.mark.slow

CORPUS = Path(__file__).resolve().parent.parent / "benchmarks" / "corpus" / "03.wav"
CHUNK = 320 * 2
TRAILING_SILENCE_CHUNKS = 60

# U+3040-309F hiragana, U+30A0-30FF katakana, U+4E00-9FFF kanji.
_JAPANESE = tuple(range(0x3040, 0x3100)) + tuple(range(0x4E00, 0xA000))


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "live.db")
    migrate(conn)
    seed(conn)
    yield conn
    conn.close()


async def test_a_real_utterance_becomes_real_speech(db: sqlite3.Connection) -> None:
    # One thread owns both models for their whole life, and loads them itself.
    # Constraint 6, and T2.6 measured why it matters: on the event loop, inference
    # blocks frame delivery for its entire duration.
    base = MlxLlm()
    worker = InferenceWorker()
    asr = OchaWhisper(sample_rate=SAMPLE_RATE, worker=worker)
    worker.start(base.load, asr.warm)
    llm = WorkerLlm(worker, base)

    # The shipped assembly minus the transport, so the two probe taps and the stage
    # order are the ones that ship. Building a bespoke chain here would measure a
    # pipeline nobody runs.
    tts = VoicevoxTTS(sample_rate=SAMPLE_RATE)
    fillers = await FillerBank.synthesise(tts, SAMPLE_RATE)
    filler_state = FillerState()
    # Emitter constructed first: it registers itself on the state, and the trigger
    # fires through it. It also has to sit LAST in the pipeline -- see filler.py.
    emitter = FillerProcessor(fillers, filler_state, emit=True)

    spans = Spans()
    timeline = TurnTimeline()
    probe = TurnStateProbe(timeline=timeline, spans=spans, emit_state=True)
    pipeline = Pipeline(
        [
            VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VAD_PARAMS)),
            TurnStateProbe(timeline=timeline, spans=spans, emit_state=True),
            FillerProcessor(fillers, filler_state),
            asr,
            TurnStateProbe(timeline=timeline, spans=spans, emit_state=True),
            TutorStage(db, ItemScheduler(db), load_grammar(), llm),
            TurnStateProbe(timeline=timeline, spans=spans, emit_state=True),
            tts,
            emitter,
            probe,
        ]
    )

    with wave.open(str(CORPUS)) as w:
        audio = w.readframes(w.getnframes())
    frames: list[Frame] = [
        InputAudioRawFrame(audio=audio[i : i + CHUNK], sample_rate=SAMPLE_RATE, num_channels=1)
        for i in range(0, len(audio), CHUNK)
    ]
    frames += [
        InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=SAMPLE_RATE, num_channels=1)
        for _ in range(TRAILING_SILENCE_CHUNKS)
    ]

    down: Sequence[Frame] = (await run_test(pipeline, frames_to_send=frames))[0]

    errors = [f.error for f in down if isinstance(f, ErrorFrame)]
    assert not errors, errors

    transcripts = [f.text for f in down if isinstance(f, TranscriptionFrame)]
    assert transcripts, "whisper produced nothing"
    assert any(ord(c) in _JAPANESE for c in transcripts[0]), f"not Japanese: {transcripts[0]!r}"

    spoken = sum(len(f.audio) for f in down if isinstance(f, TTSAudioRawFrame))
    assert spoken > 0, "no audio was synthesised"

    report = probe.report()
    print(f"\ntranscript: {transcripts[0]}")
    print(f"stages: asr {report['asr_s']}  llm {report['llm_ttft_s']}  tts {report['tts_s']}")
    print(f"synthesised: {spoken / 2 / SAMPLE_RATE:.2f}s of audio")
    print(f"report: {report}")

    # A regression bound, not a target. G1b's p50 is 3.2 s over 50 turns; a single
    # turn exceeding 8 s means something is structurally wrong -- a cold load in the
    # path, or a stage that buffered.
    v2fa = report["voice_to_first_audio_s"]
    assert isinstance(v2fa, float) and 0 < v2fa < 8.0, f"implausible: {v2fa}"

    # Filled pauses must have covered the wait. If none played, G1a's remedy is
    # not working and the number above is the only thing keeping the turn tolerable.
    assert any(isinstance(f, FillerAudioFrame) for f in down), "no filled pause played"
    worker.stop()
