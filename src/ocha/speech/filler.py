"""Filled pauses (T2.8) — pedagogically load-bearing, not latency papering.

Japanese filled pauses (ええと, あの, そうですね, なるほど, うーん) are input a
learner working from textbooks and apps does not get, and a beginner who has never
heard them defaults to English "um" when speaking. Producing them is part of
sounding like a speaker rather than a reader. So the tutor uses them, in the place
a speaker would: while thinking.

That it also removes the dead air G1a measures is a genuine second benefit, and
the distinction matters for how the criterion is judged: **this removes silence
rather than redefining silence as acceptable.** G1a's threshold is unchanged and
G1b's bound is untouched.

## Why the audio does not count toward G1b

`FillerAudioFrame` is a distinct type, and `TurnStateProbe` excludes it from the
`first_audio` mark. Counting it would make voice-to-first-audio measure how fast
the tutor can say「ええと」-- a number that improves while the product gets no
better. It does count as a state change for G1a, because the user genuinely hears
something.

## Why the trigger and the emitter are in different places

**Trigger, after VAD.** It has to see the endpoint the moment silero declares it.
`SegmentedSTTService` forwards `VADUserStoppedSpeakingFrame` only *after* it has
transcribed, so anything further down learns about the endpoint a second late --
which is exactly the gap being covered.

**Emitter, last.** This was learned by measuring. The first version pushed the
audio from the trigger's own position, and it arrived ~1.0 s late anyway: a
processor forwards frames from its queue in order, and the ASR service's queue is
blocked while it transcribes, so the filler waited behind the thing it was covering
for. The tutor stage and the TTS service block their queues the same way. The
emitter therefore sits **downstream of every stage that blocks** -- last before the
transport -- where nothing can hold its frames up.

Being last also lets the emitter see real synthesised audio go past, which is its
cancel signal, so no third instance is needed.

They share a `FillerState`. Same shape as the probe's shared `Spans`, and for the
same reason: one mechanism, several positions.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Short, neutral, and all usable by an adult male speaker in polite conversation.
# 「はい」 is excluded deliberately: it means "yes", so using it as a thinking noise
# would teach the learner to agree with things they have not understood.
FILLERS = ("ええと。", "あの。", "そうですね。", "なるほど。", "うーん。")

# How long after a filler finishes to wait before deciding another is needed.
# Below G1a's 500 ms so the second one lands inside the budget rather than at it.
FOLLOW_UP_AFTER_S = 0.35

# Two at most. A third would mean the wait is long enough that the latency is the
# problem and the filler is covering for it -- which is the thing this is not for.
MAX_PER_TURN = 2


class FillerAudioFrame(TTSAudioRawFrame):
    """Audio the tutor produced to fill a pause, not to say anything.

    A distinct type so it can be counted for G1a and excluded from G1b.
    """


@dataclass
class FillerState:
    """Shared between the trigger and the emitter."""

    real_audio: bool = False
    played: int = 0
    task: asyncio.Task[None] | None = None
    order: list[int] = field(default_factory=list)
    last: int | None = None
    # Set by the emitter when it is constructed. The trigger fires through this
    # rather than pushing itself -- see the module docstring on why position matters.
    emitter: FillerProcessor | None = None

    def reset(self) -> None:
        self.real_audio = False
        self.played = 0

    def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = None


class FillerBank:
    """Pre-synthesised PCM, so playback costs no synthesis at all.

    Built at startup, where ~0.4 s per phrase is free. Doing it per turn would put
    a VOICEVOX round trip inside the gap it is meant to cover.
    """

    def __init__(self, clips: dict[str, bytes], sample_rate: int) -> None:
        if not clips:
            raise ValueError("an empty filler bank cannot fill anything")
        self.clips = clips
        self.sample_rate = sample_rate
        self._texts = list(clips)

    @classmethod
    async def synthesise(
        cls, tts: object, sample_rate: int, texts: tuple[str, ...] = FILLERS
    ) -> FillerBank:
        """Render every filler once, through the real TTS service.

        Raises with the actual cause rather than "empty bank" -- the first time
        VOICEVOX was down, startup failed with a message about the bank being
        empty, which is the symptom and sends you to the wrong file.
        """
        if hasattr(tts, "reachable") and not tts.reachable():
            raise RuntimeError(
                "VOICEVOX is not answering on its HTTP port. It is a hard dependency: "
                "no TTS, no spoken reply, and no filled pauses. Start the engine and retry."
            )
        clips: dict[str, bytes] = {}
        for text in texts:
            audio = b""
            async for frame in tts.speak(text):  # type: ignore[attr-defined]
                if isinstance(frame, TTSAudioRawFrame):
                    audio += frame.audio
            if audio:
                clips[text] = audio
        return cls(clips, sample_rate)

    def pick(self, state: FillerState) -> tuple[str, bytes]:
        """Next filler, shuffled without immediate repeats.

        A fixed rotation is as much of a tic as a fixed choice -- the learner
        notices the cycle instead of the phrase. So the order is reshuffled each
        pass, and the reshuffle explicitly avoids starting with whatever was said
        last: shuffling alone lets the same phrase land either side of the boundary,
        which is the one case a listener is guaranteed to notice.
        """
        if not state.order:
            order = list(range(len(self._texts)))
            random.shuffle(order)
            if len(order) > 1 and order[-1] == state.last:
                order[-1], order[0] = order[0], order[-1]
            state.order = order
        index = state.order.pop()
        state.last = index
        text = self._texts[index]
        return text, self.clips[text]

    def duration(self, audio: bytes) -> float:
        return len(audio) / 2 / self.sample_rate


class FillerProcessor(FrameProcessor):
    """`emit=False` triggers at the VAD endpoint; `emit=True` does the talking."""

    # 10 ms per frame at 16 kHz, matching the TTS service's chunking.
    CHUNK_BYTES = 320

    def __init__(self, bank: FillerBank, state: FillerState, *, emit: bool = False) -> None:
        super().__init__()
        self._bank = bank
        self._state = state
        self._emit = emit
        if emit:
            state.emitter = self

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, CancelFrame | EndFrame):
            self._state.cancel()

        if self._emit:
            # Real synthesised speech has gone past, so no follow-up is wanted. What
            # was already said is not recalled -- it was heard, and it was not wrong
            # to say it.
            if isinstance(frame, TTSAudioRawFrame) and not isinstance(frame, FillerAudioFrame):
                self._state.real_audio = True
                self._state.cancel()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # Barge-in. Stop talking, including stopping thinking out loud.
            self._state.cancel()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._state.cancel()
            self._state.reset()
            emitter = self._state.emitter
            if emitter is not None:
                self._state.task = asyncio.create_task(emitter._fill())

        await self.push_frame(frame, direction)

    async def _fill(self) -> None:
        """Play a filler, then a second one if the tutor is still not talking."""
        try:
            while self._state.played < MAX_PER_TURN and not self._state.real_audio:
                text, audio = self._bank.pick(self._state)
                self._state.played += 1
                for i in range(0, len(audio), self.CHUNK_BYTES):
                    await self.push_frame(
                        FillerAudioFrame(
                            audio=audio[i : i + self.CHUNK_BYTES],
                            sample_rate=self._bank.sample_rate,
                            num_channels=1,
                        ),
                        FrameDirection.DOWNSTREAM,
                    )
                # Wait out the clip, then a little more, before deciding again.
                await asyncio.sleep(self._bank.duration(audio) + FOLLOW_UP_AFTER_S)
        except asyncio.CancelledError:
            pass  # barge-in, or real audio arrived
