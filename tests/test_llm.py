"""T1.7 — LLM service. The real model is 14.2 GB, so these test the contract."""

from __future__ import annotations

import pytest

from ocha.tutor.llm import (
    DEFAULT_MODEL,
    LlmService,
    MlxLlm,
    StubLlm,
    _prefix_key,
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


def test_status_reports_not_loaded_before_load() -> None:
    st = MlxLlm().status()
    assert st.loaded is False
    assert st.resident_gb == 0.0


# ---- KV-cache prefix keying ---------------------------------------------


def test_prefix_key_is_stable_and_discriminating() -> None:
    a = "You are a Japanese conversation partner.\nKNOWN: 私、これ"
    b = "You are a Japanese conversation partner.\nKNOWN: 私、これ、それ"
    assert _prefix_key(a) == _prefix_key(a)
    assert _prefix_key(a) != _prefix_key(b)


def test_cache_is_rebuilt_when_the_system_prompt_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The correctness issue hiding inside a latency optimisation.

    The KV cache encodes one specific prefix. The Context Builder rewrites the
    system prompt as FSRS state evolves, so reusing a cache built on prompt A
    while sending prompt B would silently feed the model the wrong context --
    a correctness bug that looks like a performance win.
    """
    import mlx_lm.models.cache as cache_mod

    made: list[object] = []

    def fake_make(_model: object) -> object:
        made.append(object())
        return made[-1]

    monkeypatch.setattr(cache_mod, "make_prompt_cache", fake_make)

    llm = MlxLlm()
    llm._model = object()  # pretend loaded; no weights involved

    first = llm._prompt_cache("PROMPT A")
    again = llm._prompt_cache("PROMPT A")
    assert again is first, "same prefix must reuse the cache (T0.4: 1.81s -> 0.50s flat)"
    assert len(made) == 1

    changed = llm._prompt_cache("PROMPT B")
    assert changed is not first, "changed prefix must NOT reuse a stale cache"
    assert len(made) == 2


# ---- stub behaviour -----------------------------------------------------


def test_stub_records_what_it_was_given() -> None:
    stub = StubLlm(reply="はい。")
    assert stub.generate("SYS", "USER") == "はい。"
    assert stub.calls == [("SYS", "USER")]


def test_stub_streams_the_same_text_it_generates() -> None:
    stub = StubLlm(reply="いいですね。")
    assert "".join(stub.stream("SYS", "USER")) == stub.generate("SYS", "USER")


def test_stream_interface_exists_for_phase_2() -> None:
    """Unused in Phase 1, required by Phase 2's sentence chunker. Building it now
    keeps Phase 2 from having to reshape the service."""
    assert hasattr(MlxLlm, "stream")
    assert hasattr(StubLlm, "stream")
