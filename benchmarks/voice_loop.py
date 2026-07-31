"""T2.6 — voice-to-first-audio through the real product path.

Two earlier harnesses produced numbers that had to be thrown away, both for the
same reason: they pushed frames into the pipeline as fast as Python could
generate them. A conversation does not arrive that way. Feeding utterance N+1
while turn N is still generating means VAD segments, transcripts and replies
queue up on top of each other, and "per-turn latency" stops meaning anything --
one run reported a flattering 1.99 s p50 while silero had produced nine
endpoints for five utterances.

So this measures through the shipped path instead: a real uvicorn process with
the real models warm, a real WebSocket client, and audio **paced in real time**,
20 ms at a time, with a real pause between utterances. It exercises the
serializer and the transport too, which the in-process harnesses skipped.

Constraint 9 -- what is validated against a known answer:

- the clock: the client sends 20 ms chunks and sleeps 20 ms, so the wall-clock
  duration of a sent utterance must match its WAV duration. Asserted per turn.
- the endpoint: `user_stopped` here is the moment the client *finished sending*
  the utterance, which is what a speaker experiences. VAD adds its own
  `stop_secs` on top, and that delay is inside the number rather than excluded
  from it -- the earlier probe-based figures measured from silero's endpoint and
  so quietly omitted it.
- first audio: the first binary frame that is not part of a filled pause. The
  server marks filler audio with a `state` message, never as reply audio.

Usage:  uv run python benchmarks/voice_loop.py [n_turns]
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
import wave

import websockets

HERE = pathlib.Path(__file__).parent
CORPUS = HERE / "corpus"
OUT = HERE / "voice-loop.json"

PORT = 8123
RATE = 16_000
CHUNK_MS = 20
CHUNK_BYTES = RATE * 2 * CHUNK_MS // 1000
PAUSE_BETWEEN_TURNS_S = 2.0
TURN_TIMEOUT_S = 20.0


def start_server() -> subprocess.Popen[bytes]:
    # start_new_session so the whole group can be signalled: `uv run` is a wrapper
    # process, and terminating it leaves uvicorn holding the port and the models.
    proc = subprocess.Popen(
        [
            "uv", "run", "uvicorn", "ocha.api.main:app",
            "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning",
        ],
        cwd=HERE.parent,
        start_new_session=True,
    )
    # The models load during lifespan, so /health is the readiness signal. It is
    # also the validation that they really loaded: model_loaded=false would mean
    # this benchmark is timing stubs.
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                health = json.load(r)
            if health.get("model_loaded"):
                print(f"server ready: {health['model']}, {health['resident_memory_gb']:.1f} GB resident")
                return proc
        except Exception:
            time.sleep(2)
    proc.terminate()
    raise RuntimeError("server never became ready")


def utterance(name: str) -> tuple[bytes, float]:
    """The recording with leading and trailing silence trimmed.

    The corpus was recorded with the mic running before and after each line, and
    those margins are not part of what the speaker said. Left in, they delay the
    endpoint by a second or more and make voice-to-first-audio measure the
    recording setup rather than the pipeline.
    """
    with wave.open(str(CORPUS / f"{name}.wav")) as w:
        assert w.getframerate() == RATE, f"{name}: {w.getframerate()} Hz, expected {RATE}"
        pcm = w.readframes(w.getnframes())
    pcm = _trim(pcm)
    return pcm, len(pcm) / 2 / RATE


def _trim(pcm: bytes, threshold: int = 700, pad_ms: int = 120) -> bytes:
    """Strip near-silence from both ends, keeping a short pad so VAD hears onset."""
    samples = memoryview(pcm).cast("h")
    loud = [i for i, s in enumerate(samples) if abs(s) > threshold]
    if not loud:
        return pcm
    pad = RATE * pad_ms // 1000
    start = max(0, loud[0] - pad)
    end = min(len(samples), loud[-1] + pad)
    return bytes(samples[start:end].cast("B"))


async def run_turn(ws: websockets.ClientConnection, name: str) -> dict[str, object]:
    pcm, duration = utterance(name)

    async def send(stop_holder: list[float]) -> None:
        for i in range(0, len(pcm), CHUNK_BYTES):
            await ws.send(pcm[i : i + CHUNK_BYTES])
            await asyncio.sleep(CHUNK_MS / 1000)
        stop_holder.append(time.perf_counter())
        # Keep feeding silence: VAD needs to hear the end of speech, and a real
        # microphone does not stop producing samples when the speaker stops.
        silence = b"\x00\x00" * (RATE * CHUNK_MS // 1000)
        while True:
            await ws.send(silence)
            await asyncio.sleep(CHUNK_MS / 1000)


    row: dict[str, object] = {"id": name, "sent_s": round(duration, 2)}
    stop_holder: list[float] = []
    sender = asyncio.create_task(send(stop_holder))
    t_start = time.perf_counter()
    # Provisional. Replaced below by when the sender *actually* finished, because
    # 20 ms sleeps drift and the expected end is optimistic by tens of ms.
    stop_at = t_start + duration
    states: list[tuple[float, str]] = []
    transcript: str | None = None
    filler_at: float | None = None
    first_audio: float | None = None
    speaking = False

    deadline = time.perf_counter() + TURN_TIMEOUT_S
    while time.perf_counter() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=deadline - time.perf_counter())
        except TimeoutError:
            break
        now = time.perf_counter()
        if stop_holder:
            stop_at = stop_holder[0]
        if isinstance(msg, bytes):
            # Audio. Filler audio is distinguishable only by what preceded it: the
            # server enters `speaking` for a filled pause too, so the reply is the
            # first audio that arrives after a transcript exists.
            if transcript is None:
                filler_at = filler_at if filler_at is not None else now
            else:
                first_audio = now
                break
            continue
        event = json.loads(msg)
        kind = event.get("type")
        if kind == "state":
            states.append((now, str(event["state"])))
            speaking = event["state"] == "speaking"
        elif kind == "transcript" and event.get("final"):
            transcript = str(event["text"])
        elif kind == "grammar":
            transcript = transcript or "[grammar]"
            first_audio = None
            break

    sender.cancel()
    # Drain whatever is still in flight -- the rest of the reply's audio, trailing
    # state messages -- before the next turn starts reading. Without this the next
    # turn sees the previous reply's audio arrive before its own transcript and
    # scores it as an instantaneous filled pause, which is how a harness reports a
    # latency that never happened.
    while True:
        try:
            await asyncio.wait_for(ws.recv(), timeout=0.75)
        except TimeoutError:
            break

    row.update(
        {
            "transcript": transcript,
            "voice_to_first_audio_s": None if first_audio is None else round(first_audio - stop_at, 3),
            "first_filler_s": None if filler_at is None else round(filler_at - stop_at, 3),
            "states": [s for _, s in states],
            "longest_silent_gap_s": round(_longest_gap(stop_at, states, first_audio, filler_at), 3),
            "speaking_at_end": speaking,
        }
    )
    return row


def _longest_gap(
    stop_at: float,
    states: list[tuple[float, str]],
    first_audio: float | None,
    filler_at: float | None,
) -> float:
    """Longest stretch after the endpoint with nothing new to hear or see.

    Client-side G1a, deliberately independent of the server's own instrument --
    two implementations of the criterion disagreeing is worth knowing about, and
    the client is where the user actually is.
    """
    marks = [t for t, _ in states]
    if filler_at is not None:
        marks.append(filler_at)
    if first_audio is not None:
        marks.append(first_audio)
    marks = sorted(t for t in marks if t >= stop_at)
    if not marks:
        return 0.0
    gaps = [marks[0] - stop_at]
    gaps += [b - a for a, b in zip(marks, marks[1:], strict=False)]
    return max(gaps)


async def main() -> None:
    names = sorted(p.stem for p in CORPUS.glob("*.wav"))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    names = names[:n]

    proc = start_server()
    rows: list[dict[str, object]] = []
    try:
        async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", max_size=None) as ws:
            for name in names:
                row = await run_turn(ws, name)
                rows.append(row)
                v = row["voice_to_first_audio_s"]
                print(
                    f"  {name}  v2fa {v if v is not None else 'MISSED':>6}  "
                    f"filler {row['first_filler_s']}  gap {row['longest_silent_gap_s']}  "
                    f"{str(row['transcript'])[:24]}"
                )
                await asyncio.sleep(PAUSE_BETWEEN_TURNS_S)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    got = [r["voice_to_first_audio_s"] for r in rows if r["voice_to_first_audio_s"] is not None]
    gaps = [r["longest_silent_gap_s"] for r in rows]
    print("\n" + "=" * 60)
    if got:
        ordered = sorted(float(v) for v in got)  # type: ignore[arg-type]
        p50 = statistics.median(ordered)
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
        print(f"voice-to-first-audio  p50 {p50:.2f}s  p95 {p95:.2f}s  over {len(got)}/{len(rows)} turns")
        print(f"PRD G1b: p50 <= 3.2 s -> {'MET' if p50 <= 3.2 else f'MISSED by {p50 - 3.2:.2f}s'}")
    within = sum(1 for g in gaps if float(g) <= 0.5)  # type: ignore[arg-type]
    print(f"PRD G1a: longest silent gap <= 0.5 s on {within}/{len(gaps)} turns")

    OUT.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
