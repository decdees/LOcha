"""The voice pipeline (T2.1--T2.5).

    transport.input() -> VAD -> ASR -> TutorStage -> TTS -> probe -> output()

The probe sits immediately before `transport.output()`, one instance, not two.
Frames flow downstream through the whole chain, so a single tap at the end sees
every stage boundary §5.1 budgets, and it is the only position from which it can
push client state messages that reach the transport without crossing back
upstream.

**There is no separate sentence chunker (T2.4).** Pipecat's `TTSService`
aggregates text into sentences before synthesising, and its boundary set already
includes 。！？. Writing our own splitter to feed it would be a second
implementation of the same rule, so T2.4 is the `TutorStage` emitting one
`LLMTextFrame` per sentence plus that default -- configuration, not code.

`build_loopback` is kept: it is the day-one audio test, and the only way to
answer by ear whether HFP-duplex headset audio is usable (ARCHITECTURE risk 4)
without ASR in the way. `/ws?loopback=1` selects it.
"""

from __future__ import annotations

import sqlite3
from typing import Literal

from fastapi import WebSocket
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from ocha.scheduling.scheduler import ItemScheduler
from ocha.speech.asr import OchaWhisper
from ocha.speech.attribution import (
    AttributionInputProcessor,
    AttributionState,
    ExchangeEndpointProcessor,
    OutputAttributionProcessor,
)
from ocha.speech.filler import FillerBank, FillerProcessor, FillerState
from ocha.speech.guided_stage import GuidedLessonStage
from ocha.speech.probe import Spans, TurnStateProbe
from ocha.speech.repair import AsrRepairProcessor
from ocha.speech.tts import VoicevoxTTS
from ocha.speech.tutor_stage import TutorStage
from ocha.speech.wire import CHANNELS, SAMPLE_RATE, ClientText, OchaSerializer
from ocha.turnstate import TurnTimeline
from ocha.tutor.grammar import GrammarReference
from ocha.tutor.llm import LlmService

# stop_secs is the endpoint delay -- how long silence must last before the turn is
# considered over. ARCHITECTURE §5.1 budgets 150 ms and calls it unmeasured; the
# first implementation used Pipecat's 0.2 s default.
#
# **0.2 s cuts the learner off.** Measured end to end (benchmarks/voice_loop.py):
# it endpointed mid-utterance on 6 of 8 corpus recordings, producing transcripts
# like 「が」 and 「コーヒーを」 -- the tutor answering a fragment while the user was
# still speaking. That is not a latency problem, it is the product interrupting a
# beginner, and beginners pause mid-sentence precisely because they are beginners.
#
# 0.6 s is a deliberate trade, not a tuned number: it sits **inside**
# voice-to-first-audio, so G1b pays for it directly. The principled fix is
# Pipecat's smart-turn model (ARCHITECTURE §2 lists it alongside silero and it has
# never been wired up), which decides whether an utterance is *finished* rather
# than whether the room is quiet. Until then, being slower is better than
# interrupting.
VAD_PARAMS = VADParams(stop_secs=0.6)


def build_transport(
    websocket: WebSocket, attribution: AttributionState | None = None
) -> FastAPIWebsocketTransport:
    return FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            audio_in_channels=CHANNELS,
            audio_out_channels=CHANNELS,
            # No WAV header: the client is ours and reads raw PCM. A header per
            # chunk would be 44 bytes of nothing on every 10 ms of audio.
            add_wav_header=False,
            serializer=OchaSerializer(attribution),
        ),
    )


class _Loopback(FrameProcessor):
    """Turns captured audio back into playable audio. Diagnostic only.

    Needed because input and output audio are distinct frame types, and the
    serializer deliberately only writes `OutputAudioRawFrame`: `InputAudioRawFrame`
    flows the length of the real pipeline too, and serializing it as well would
    echo the user's own microphone back at them.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            frame = OutputAudioRawFrame(
                audio=frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
        await self.push_frame(frame, direction)


def build_loopback(
    transport: FastAPIWebsocketTransport, attribution: AttributionState | None = None
) -> tuple[Pipeline, TurnStateProbe]:
    """Audio in, same audio out. No VAD, no models. The by-ear audio check."""
    probe = TurnStateProbe(emit_state=True)
    attribution = attribution or AttributionState()
    return (
        Pipeline(
            [
                transport.input(),
                AttributionInputProcessor(attribution),
                _Loopback(),
                probe,
                OutputAttributionProcessor(attribution),
                transport.output(),
            ]
        ),
        probe,
    )


def build_pipeline(
    transport: FastAPIWebsocketTransport,
    conn: sqlite3.Connection,
    scheduler: ItemScheduler,
    reference: GrammarReference,
    llm: LlmService,
    *,
    asr: SegmentedSTTService | None = None,
    tts: FrameProcessor | None = None,
    fillers: FillerBank | None = None,
    repair_audio: bytes | None = None,
    attribution: AttributionState | None = None,
    mode: Literal["guided", "conversation"] = "conversation",
) -> tuple[Pipeline, TurnStateProbe]:
    """The real loop. `asr`/`tts` are injectable so tests can run it without MLX.

    **Four probes, one instrument.** They share a `Spans` and a `TurnTimeline`;
    the returned probe is the tail one, which therefore reports for all three. Each
    position is there because a stage downstream of it delays the frame the
    measurement depends on:

    - **after VAD** -- `SegmentedSTTService` forwards `VADUserStoppedSpeakingFrame`
      only *after* it has transcribed the segment, so any later tap timestamps the
      end of the utterance at the end of ASR. That makes `asr_s` read ~0 and, worse,
      drops ASR out of `voice_to_first_audio_s` entirely, under-reporting G1b.
    - **after ASR** -- `TutorStage` blocks the event loop for the whole generation,
      so a tail-only tap sees the transcript after the LLM has finished and charges
      the LLM's time to ASR.
    - **after the tutor stage** -- `TTSService` consumes `LLMTextFrame` and emits its
      own text frame only once a sentence has been synthesised, so without this tap
      VOICEVOX's synthesis time is charged to the LLM and `tts_s` reads ~0.
    - **at the end** -- where audio actually leaves.

    All three were measured, not reasoned about: see benchmarks/voice-loop.md.
    """
    spans = Spans()
    timeline = TurnTimeline()
    attribution = attribution or AttributionState()
    taps = [TurnStateProbe(timeline=timeline, spans=spans, emit_state=True) for _ in range(4)]
    after_vad, after_asr, after_tutor, tail = taps

    # The filled-pause pair: trigger right after VAD, emitter last. The emitter is
    # last because a processor's queue is blocked while that processor is busy, so
    # audio pushed from before ASR waits behind transcription -- measured at ~1.0 s
    # late, covering nothing. See speech/filler.py. Skipped entirely when no bank
    # was synthesised: a filler that must be synthesised on demand would sit inside
    # the gap it exists to cover.
    #
    # NOTE the construction order -- the emitter registers itself on the state, so it
    # must exist before the trigger can fire through it.
    filler_state = FillerState()
    use_fillers = fillers is not None and mode == "conversation"
    emit: list[FrameProcessor] = []
    trigger: list[FrameProcessor] = []
    if use_fillers:
        assert fillers is not None
        emit.append(FillerProcessor(fillers, filler_state, emit=True))
        trigger.append(FillerProcessor(fillers, filler_state))
    tutor: FrameProcessor = (
        GuidedLessonStage(conn) if mode == "guided" else TutorStage(conn, scheduler, reference, llm)
    )

    return (
        Pipeline(
            [
                transport.input(),
                AttributionInputProcessor(attribution),
                VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VAD_PARAMS)),
                ExchangeEndpointProcessor(attribution),
                after_vad,
                *trigger,
                asr if asr is not None else OchaWhisper(),
                after_asr,
                AsrRepairProcessor(repair_audio, SAMPLE_RATE),
                tutor,
                after_tutor,
                tts if tts is not None else VoicevoxTTS(),
                *emit,
                tail,
                ClientText(),
                OutputAttributionProcessor(attribution),
                transport.output(),
            ]
        ),
        tail,
    )


async def run_session(
    websocket: WebSocket,
    conn: sqlite3.Connection,
    scheduler: ItemScheduler,
    reference: GrammarReference,
    llm: LlmService,
    *,
    loopback: bool = False,
    asr: SegmentedSTTService | None = None,
    tts: FrameProcessor | None = None,
    fillers: FillerBank | None = None,
    repair_audio: bytes | None = None,
    mode: Literal["guided", "conversation"] = "conversation",
) -> TurnStateProbe:
    """Serve one client connection for its whole life. Returns the probe.

    The probe is returned rather than logged so the caller decides what to do
    with the measurement -- T2.6 writes it to a report, tests assert on it.
    """
    attribution = AttributionState()
    transport = build_transport(websocket, attribution)
    if loopback:
        pipeline, probe = build_loopback(transport, attribution)
    else:
        pipeline, probe = build_pipeline(
            transport,
            conn,
            scheduler,
            reference,
            llm,
            asr=asr,
            tts=tts,
            fillers=fillers,
            repair_audio=repair_audio,
            attribution=attribution,
            mode=mode,
        )
    # PipelineWorker/WorkerRunner, not PipelineTask/PipelineRunner: the latter pair
    # is deprecated as of Pipecat 1.3 and removed at 2.0. Same objects, new names.
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE),
        # RTVI is Pipecat's own client protocol, on by default. We do not speak it
        # -- the client reads `speech/wire.py`'s handful of message types -- and
        # left on it puts ~15 extra JSON messages per turn on the socket
        # (bot-llm-started, bot-transcription, bot-tts-stopped...). Observed in a
        # browser, where they arrived alongside ours and were silently ignored.
        enable_rtvi=False,
    )
    # handle_sigint=False: uvicorn owns the signal handlers. A runner that installs
    # its own turns Ctrl-C into a hung server.
    runner = WorkerRunner(handle_sigint=False)
    # `add_workers` is a coroutine despite the name. Calling it without await is a
    # silent no-op -- the runner then starts with zero workers and blocks forever,
    # which presents as a transport hang rather than an error.
    await runner.add_workers(worker)
    # auto_end (the default) ends the runner when the pipeline ends, which for one
    # runner per connection is exactly the client disconnecting.
    await runner.run()
    attribution.finalize()
    return probe
