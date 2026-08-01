"""T1.7 — LLM service. The real model is 14.2 GB, so these test the contract."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

import ocha.tutor.llm as llm_module
from ocha.tutor.llm import (
    DEFAULT_MODEL,
    ChatMessage,
    LlmService,
    MlxLlm,
    StubLlm,
    model_id,
)

# ---- contract ------------------------------------------------------------


def test_stub_and_real_satisfy_the_same_protocol() -> None:
    assert isinstance(StubLlm(), LlmService)
    assert isinstance(MlxLlm(), LlmService)


def test_default_model_is_the_one_decision_md_names() -> None:
    assert DEFAULT_MODEL == "mlx-community/Qwen3.5-9B-4bit"


def test_model_is_a_config_value_not_a_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """DECISION.md: swapping the model is a config edit, not a code change."""
    monkeypatch.setenv("OCHA_LLM_MODEL", "some/other-model")
    assert model_id() == "some/other-model"
    assert MlxLlm().model_name == "some/other-model"


# ---- no silent fallback --------------------------------------------------


def test_generate_before_load_raises_rather_than_loading() -> None:
    """Lazy-loading on first request would move a measured 6-7 s cold load into a
    request path -- project standing constraint 3 calls that a correctness bug."""
    llm = MlxLlm()
    with pytest.raises(RuntimeError, match="not loaded"):
        llm.generate("system", "user")


def test_missing_model_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no cloud fallback, so an unavailable model is fatal, not degraded."""
    monkeypatch.setenv("OCHA_LLM_MODEL", "definitely/not-a-real-model-xyz")
    llm = MlxLlm()
    with pytest.raises(RuntimeError, match="no cloud fallback"):
        llm.load()


def test_load_passes_a_resolved_local_path_to_mlx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    fake_mlx_lm = types.ModuleType("mlx_lm")

    def fake_load(path: str) -> tuple[object, object]:
        seen.append(path)
        return object(), object()

    fake_mlx_lm.load = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setattr(llm_module, "resolve_cached_model", lambda _repo: tmp_path)

    MlxLlm().load()
    assert seen == [str(tmp_path)]


def test_status_reports_not_loaded_before_load() -> None:
    st = MlxLlm().status()
    assert st.loaded is False
    assert st.resident_gb == 0.0


# ---- explicit bounded history, without mutable inference state ----------


class CaptureTokenizer:
    def __init__(self) -> None:
        self.rendered_messages: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> str:
        self.rendered_messages.append(messages)
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    def encode(self, text: str) -> list[str]:
        return list(text)


def test_history_is_role_tagged_and_oldest_complete_exchanges_are_dropped() -> None:
    llm = MlxLlm()
    tok = CaptureTokenizer()
    llm._tok = tok
    history = tuple(
        message
        for n in range(5)
        for message in (
            ChatMessage("user", f"u{n}" + "x" * 300),
            ChatMessage("assistant", f"a{n}" + "y" * 300),
        )
    )

    rendered = llm._render("system", history, "current")

    final = tok.rendered_messages[-1]
    assert final[0] == {"role": "system", "content": "system"}
    assert [m["role"] for m in final].count("system") == 1
    assert final[-1] == {"role": "user", "content": "current"}
    assert all("u0" not in m["content"] and "a0" not in m["content"] for m in final)
    assert [m["role"] for m in final[1:-1]] == ["user", "assistant"] * 3
    assert len(tok.encode(rendered)) <= 2048


def test_system_and_current_turn_over_budget_fail_explicitly() -> None:
    llm = MlxLlm()
    llm._tok = CaptureTokenizer()
    with pytest.raises(ValueError, match="system prompt plus current turn"):
        llm._render("s" * 2048, (), "current")


def test_stream_does_not_create_or_pass_a_mutable_prompt_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    fake_mlx_lm = types.ModuleType("mlx_lm")

    def fake_stream_generate(*args: object, **kwargs: object) -> list[object]:
        seen.update(kwargs)
        return [types.SimpleNamespace(text="はい。")]

    fake_mlx_lm.stream_generate = fake_stream_generate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    llm = MlxLlm()
    llm._model = object()
    llm._tok = CaptureTokenizer()

    assert "".join(llm.stream("system", "user")) == "はい。"
    assert "prompt_cache" not in seen
    assert not hasattr(llm, "_cache")
    assert not hasattr(llm, "_cache_key")


# ---- stub behaviour -----------------------------------------------------


def test_stub_records_what_it_was_given() -> None:
    stub = StubLlm(reply="はい。")
    assert stub.generate("SYS", "USER") == (
        '{"japanese": "はい。", "english": "Test English meaning"}'
    )
    assert stub.calls == [("SYS", "USER")]


def test_stub_accepts_explicit_history() -> None:
    stub = StubLlm(reply="はい。")
    history = (ChatMessage("user", "前"), ChatMessage("assistant", "返答"))
    assert stub.generate("SYS", "USER", history=history) == (
        '{"japanese": "はい。", "english": "Test English meaning"}'
    )
    assert stub.histories == [history]


def test_stub_streams_the_same_text_it_generates() -> None:
    stub = StubLlm(reply="いいですね。")
    assert "".join(stub.stream("SYS", "USER")) == stub.generate("SYS", "USER")


def test_stream_interface_exists_for_phase_2() -> None:
    """Unused in Phase 1, required by Phase 2's sentence chunker. Building it now
    keeps Phase 2 from having to reshape the service."""
    assert hasattr(MlxLlm, "stream")
    assert hasattr(StubLlm, "stream")
