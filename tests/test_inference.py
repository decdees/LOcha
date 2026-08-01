"""Dedicated inference worker ownership and lifecycle guarantees."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from ocha.inference import InferenceWorker, WorkerLlm
from ocha.tutor.llm import ChatMessage, LlmStatus


class ThreadRecordingLlm:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        *,
        history: tuple[ChatMessage, ...] = (),
        max_tokens: int = 64,
    ) -> str:
        self.thread_ids.append(threading.get_ident())
        return "はい。"

    def stream(
        self,
        system_prompt: str,
        user_text: str,
        *,
        history: tuple[ChatMessage, ...] = (),
        max_tokens: int = 64,
    ) -> Iterator[str]:
        self.thread_ids.append(threading.get_ident())
        yield "はい。"

    def status(self) -> LlmStatus:
        self.thread_ids.append(threading.get_ident())
        return LlmStatus(model="test", loaded=True, resident_gb=0.0)


def test_generation_and_status_share_the_worker_thread() -> None:
    worker = InferenceWorker()
    base = ThreadRecordingLlm()
    worker.start()
    try:
        llm = WorkerLlm(worker, base)
        history = (ChatMessage("user", "前"), ChatMessage("assistant", "返答"))
        assert llm.generate("system", "user", history=history) == "はい。"
        assert llm.status().loaded
        assert base.thread_ids == [worker.thread_id, worker.thread_id]
    finally:
        worker.stop()


async def test_async_full_generation_stays_on_the_worker_thread() -> None:
    worker = InferenceWorker()
    base = ThreadRecordingLlm()
    worker.start()
    try:
        llm = WorkerLlm(worker, base)
        assert await llm.agenerate("system", "user") == "はい。"
        assert base.thread_ids == [worker.thread_id]
    finally:
        worker.stop()


def test_loader_failure_stops_and_joins_the_worker() -> None:
    worker = InferenceWorker()

    def fail() -> None:
        raise RuntimeError("load failed")

    with pytest.raises(RuntimeError, match="load failed"):
        worker.start(fail)
    assert not worker.alive
    assert not worker._thread.is_alive()
