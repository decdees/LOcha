"""The inference worker — one thread, one queue, both models for its whole life.

This is the shape standing constraint 6 prescribes, and it exists because T2.6
measured why it is needed. MLX GPU streams are thread-local to the thread that ran
`load()`, so inference had been running on the event-loop thread. That is correct
and it is slow for a reason that has nothing to do with the models: for the ~1.4 s
a generation takes, the event loop cannot run the task that drains a downstream
processor's queue, so a sentence pushed to TTS at 1.5 s is not *delivered* until
generation ends, and the audio VOICEVOX synthesised in its own thread then waits
too. Measured cost: 0.6-0.9 s per turn. See `benchmarks/voice-loop.md`.

**Not `run_in_executor`, not a `ThreadPoolExecutor`, deliberately.** A pool with
one worker would satisfy the letter of the constraint today and break the moment
someone widens it, and `run_in_executor` is named in constraint 6 as the thing not
to do. An explicit thread makes the invariant checkable: `_thread` is created once
and every model call runs on it.

## The invariant

Both models are loaded **on this thread**, by `start()`, and every subsequent call
into them is submitted to the same thread. Nothing else may touch them. There is no
second worker and no fallback path that runs inference inline -- that would be a
thread-affinity bug that only appears under load.

## What does NOT belong here

The SQLite connection. It is bound to the thread that opened it (the event loop's,
during lifespan) and passing DB work to this thread reproduces the T1.8 bug from
the other direction. Model calls come here; persistence stays on the loop.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

# Sentinel closing a streamed job. A distinct object rather than None, because a
# generator is allowed to yield None.
_DONE = object()


@dataclass(slots=True)
class _Job:
    fn: Callable[[], Any]
    future: Future[Any]
    # Set for streaming jobs: chunks are handed back as they are produced instead
    # of accumulated, which is the entire point of the worker.
    sink: Callable[[Any], None] | None = None


class InferenceWorker:
    """Runs callables on one dedicated thread, in submission order."""

    def __init__(self, name: str = "ocha-inference") -> None:
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False
        self.thread_id: int | None = None

    # ---- lifecycle -----------------------------------------------------

    def start(self, *loaders: Callable[[], Any]) -> None:
        """Start the thread and load the models on it, blocking until done.

        Blocking is correct: the process must not serve a request before the
        models are warm (constraint 3), and this runs during lifespan startup
        where there is nothing to serve yet.
        """
        if self._started:
            raise RuntimeError("worker already started")
        self._started = True
        self._thread.start()
        for loader in loaders:
            self.submit(loader).result()

    def stop(self) -> None:
        if self._started:
            self._queue.put(None)
            self._thread.join(timeout=10.0)
            self._started = False

    @property
    def alive(self) -> bool:
        return self._started and self._thread.is_alive()

    # ---- submission ----------------------------------------------------

    def submit(self, fn: Callable[[], Any]) -> Future[Any]:
        """Queue `fn` for the worker thread. Returns a plain concurrent Future."""
        if not self._started:
            raise RuntimeError("worker not started -- call start() during startup")
        future: Future[Any] = Future()
        self._queue.put(_Job(fn=fn, future=future))
        return future

    def call_sync(self, fn: Callable[[], Any]) -> Any:
        """Run on the worker and block until it returns.

        For callers that are themselves synchronous -- `POST /turn` through
        `run_turn`. It blocks the calling thread, which is exactly what that
        endpoint already did; the difference is *which* thread owns the model.
        """
        return self.submit(fn).result()

    async def call(self, fn: Callable[[], Any]) -> Any:
        """Await a single result without blocking the event loop."""
        return await asyncio.wrap_future(self.submit(fn))

    async def stream(self, fn: Callable[[], Iterator[Any]]) -> Any:
        """Run a generator on the worker, yielding items to the event loop as
        they are produced.

        This is the method that recovers the latency. The worker generates while
        the loop stays free to deliver frames, so TTS synthesis for sentence one
        overlaps generation of sentence two.
        """
        loop = asyncio.get_running_loop()
        out: asyncio.Queue[Any] = asyncio.Queue()

        def sink(item: Any) -> None:
            # From the worker thread. call_soon_threadsafe is the only safe way to
            # touch loop-owned objects from another thread.
            loop.call_soon_threadsafe(out.put_nowait, item)

        future = self.submit(lambda: self._drain(fn, sink))
        awaitable = asyncio.wrap_future(future)

        while True:
            item = await out.get()
            if item is _DONE:
                break
            yield item
        # Surfaces an exception raised inside the generator, which would otherwise
        # be swallowed once _DONE arrived.
        await awaitable

    # ---- worker thread -------------------------------------------------

    @staticmethod
    def _drain(fn: Callable[[], Iterator[Any]], sink: Callable[[Any], None]) -> None:
        try:
            for item in fn():
                sink(item)
        finally:
            # In a finally block so a generator that raises still releases the
            # consumer; the exception is re-raised through the future.
            sink(_DONE)

    def _run(self) -> None:
        self.thread_id = threading.get_ident()
        while True:
            job = self._queue.get()
            if job is None:
                return
            if job.future.set_running_or_notify_cancel():
                try:
                    job.future.set_result(job.fn())
                except BaseException as exc:  # noqa: BLE001 -- relayed, not swallowed
                    job.future.set_exception(exc)


class WorkerLlm:
    """An `LlmService` whose every model call happens on the worker thread.

    A wrapper rather than a change to `MlxLlm`: the model class should not know
    about threading, and `POST /turn` needs the synchronous `LlmService` shape
    while the pipeline needs the async streaming one.
    """

    def __init__(self, worker: InferenceWorker, llm: Any) -> None:
        self._worker = worker
        self._llm = llm

    def generate(self, system_prompt: str, user_text: str, *, max_tokens: int = 64) -> str:
        return str(
            self._worker.call_sync(
                lambda: self._llm.generate(system_prompt, user_text, max_tokens=max_tokens)
            )
        )

    def stream(self, system_prompt: str, user_text: str, *, max_tokens: int = 64) -> Iterator[str]:
        """The synchronous protocol method. Collects, because a sync iterator
        cannot stream off another thread without blocking this one anyway.

        The pipeline does not use this -- it uses `astream`, which is the whole
        point of the worker. This exists so `WorkerLlm` still satisfies
        `LlmService` for callers that are not async.
        """
        return iter([self.generate(system_prompt, user_text, max_tokens=max_tokens)])

    def astream(self, system_prompt: str, user_text: str, *, max_tokens: int = 64) -> Any:
        """Token chunks, delivered to the event loop as the worker produces them."""
        return self._worker.stream(
            lambda: self._llm.stream(system_prompt, user_text, max_tokens=max_tokens)
        )

    def status(self) -> Any:
        return self._llm.status()
