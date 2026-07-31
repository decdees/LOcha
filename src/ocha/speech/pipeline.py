"""The voice pipeline (T2.1 scaffold).

What exists here today is the **loopback**: audio from the client goes through
the probe and straight back out. That is deliberately not a placeholder chain
with the real stages stubbed -- it is the day-one audio quality test TASKS.md
T2.1 asks for, and it answers questions nothing downstream can:

- does 16 kHz PCM survive the round trip over Tailscale intact
- what does HFP-duplex headset audio actually sound like (ARCHITECTURE risk 4)
- does the client's capture-and-playback path work at all

T2.2--T2.5 replace the loopback with VAD -> ASR -> LLM -> chunker -> TTS. The
probe placement and the transport do not change when they do.

The probe sits immediately before `transport.output()`, one instance, not two.
Frames flow downstream through the whole chain, so a single tap at the end sees
every stage boundary §5.1 budgets, and it is the only position from which it can
push client state messages that reach the transport without crossing back
upstream.
"""

from __future__ import annotations

from fastapi import WebSocket
from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from ocha.speech.probe import TurnStateProbe
from ocha.speech.wire import CHANNELS, SAMPLE_RATE, OchaSerializer


def build_transport(websocket: WebSocket) -> FastAPIWebsocketTransport:
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
            serializer=OchaSerializer(),
        ),
    )


class _Loopback(FrameProcessor):
    """Turns captured audio back into playable audio. Deleted at T2.2.

    Needed because input and output audio are distinct frame types, and the
    serializer deliberately only writes `OutputAudioRawFrame` -- once real stages
    exist, `InputAudioRawFrame` still flows the length of the pipeline, and
    serializing it too would echo the user's own microphone back at them.
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


def build_pipeline(transport: FastAPIWebsocketTransport) -> tuple[Pipeline, TurnStateProbe]:
    probe = TurnStateProbe(emit_state=True)
    return Pipeline([transport.input(), _Loopback(), probe, transport.output()]), probe


async def run_session(websocket: WebSocket) -> TurnStateProbe:
    """Serve one client connection for its whole life. Returns the probe.

    The probe is returned rather than logged so the caller decides what to do
    with the measurement -- T2.6 writes it to a report, tests assert on it.
    """
    transport = build_transport(websocket)
    pipeline, probe = build_pipeline(transport)
    # PipelineWorker/WorkerRunner, not PipelineTask/PipelineRunner: the latter pair
    # is deprecated as of Pipecat 1.3 and removed at 2.0. Same objects, new names.
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE),
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
    return probe
