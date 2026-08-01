"""LLM service (T1.7).

Model chosen by benchmarks/DECISION.md. Three requirements here are not
preferences -- each came out of Phase 0 measurement:

1. Loaded once at startup and kept warm. A cold MLX load is 6-7 s measured
   (T0.4); paying that in a request path is a correctness bug, not a slow path,
   because it destroys the conversational illusion (project standing constraint 3).

2. enable_thinking=False. Both candidates are reasoning models. With the channel
   on, T0.5 observed Gemma answering a grammar question with the sentinel plus
   399 characters of explanation -- the firewall discards it, but the model is
   spending hundreds of tokens leaking what FR-5 forbids. Off is required.

3. Conversation history is explicit and bounded. Mutable prompt caches previously
   made correctness depend on hidden model state; accuracy-first operation renders
   the complete bounded prompt for every turn.

The model id is a config value, not a hardcoded constant, so swapping it is a
config edit rather than a code change (DECISION.md). No plugin interface -- that
would be a speculative abstraction with one implementation.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from ocha.models import resolve_cached_model
from ocha.tutor.context import MAX_CONTEXT_TOKENS

# benchmarks/DECISION.md. Override with OCHA_LLM_MODEL.
#
# Qwen3.5-9B, not Gemma 4 26B-A4B. Gemma is 1.9x faster but produced
# 「今日は何を食べるですか？」-- です cannot attach to a plain-form verb; the
# polite interrogative is 食べますか, or 食べるんですか / 食べるのですか. That is a
# chapter-3 error, and a tutor teaching it to a beginner who cannot detect it is
# worse than a slower tutor that is correct. Gemma's throughput was bought to
# meet a latency target that ASR misses by 5x anyway -- wrong bottleneck.
DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-4bit"
MAX_REPLY_TOKENS = 64


def model_id() -> str:
    return os.environ.get("OCHA_LLM_MODEL", DEFAULT_MODEL)


@dataclass(frozen=True, slots=True)
class LlmStatus:
    model: str
    loaded: bool
    resident_gb: float


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@runtime_checkable
class LlmService(Protocol):
    """What the tutor layer depends on. Kept narrow so tests can stub it -- the
    real model is 14.2 GB and cannot live in a unit-test suite."""

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        *,
        history: Sequence[ChatMessage] = ...,
        max_tokens: int = ...,
    ) -> str: ...

    def stream(
        self,
        system_prompt: str,
        user_text: str,
        *,
        history: Sequence[ChatMessage] = ...,
        max_tokens: int = ...,
    ) -> Iterator[str]:
        """Unused in Phase 1, required in Phase 2's sentence chunker."""
        ...

    def status(self) -> LlmStatus: ...


class MlxLlm:
    """MLX-backed local inference. No cloud path exists in this class."""

    def __init__(self, model: str | None = None) -> None:
        self.model_name = model or model_id()
        # Any, not object: mlx_lm is untyped, so a narrower annotation would only
        # force casts at every use site without buying real checking.
        self._model: Any = None
        self._tok: Any = None

    # ---- lifecycle -----------------------------------------------------

    def load(self) -> None:
        """Load once. Raises loudly if unavailable -- there is no fallback,
        because there is no cloud fallback to fall back to."""
        if self._model is not None:
            return
        try:
            local_path = resolve_cached_model(self.model_name)
            from mlx_lm import load

            # load() returns (model, tokenizer) or (model, tokenizer, config)
            # depending on its return_config flag, typed as a Union. A 2-tuple
            # unpack is correct at runtime with the default but does not
            # type-check, so take the first two by index -- valid for both arities
            # and it keeps working if a future version always returns the config.
            loaded = load(str(local_path))
            self._model, self._tok = loaded[0], loaded[1]
        except Exception as exc:
            raise RuntimeError(
                f"could not load {self.model_name!r}. Ocha has no cloud fallback, so "
                f"this is fatal. Check the model is present in the local HF cache."
            ) from exc

    def status(self) -> LlmStatus:
        resident = 0.0
        if self._model is not None:
            import mlx.core as mx

            resident = round(mx.get_active_memory() / 1e9, 2)
        return LlmStatus(
            model=self.model_name,
            loaded=self._model is not None,
            resident_gb=resident,
        )

    # ---- generation ----------------------------------------------------

    def _render(
        self,
        system_prompt: str,
        history: Sequence[ChatMessage],
        user_text: str,
    ) -> str:
        assert self._tok is not None

        retained = list(history[-8:])
        if len(retained) % 2 or any(
            message.role != ("user" if index % 2 == 0 else "assistant")
            for index, message in enumerate(retained)
        ):
            raise ValueError("history must contain complete user/assistant exchanges")

        def render(messages: Sequence[ChatMessage]) -> str:
            payload = [
                {"role": "system", "content": system_prompt},
                *({"role": message.role, "content": message.content} for message in messages),
                {"role": "user", "content": user_text},
            ]
            value: str = self._tok.apply_chat_template(
                payload,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            return value

        base = render(())
        if len(self._tok.encode(base)) > MAX_CONTEXT_TOKENS:
            raise ValueError(
                "system prompt plus current turn exceeds the 2,048-token context limit"
            )

        rendered = render(retained)
        while len(self._tok.encode(rendered)) > MAX_CONTEXT_TOKENS:
            retained = retained[2:]
            rendered = render(retained)
        return rendered

    def stream(
        self,
        system_prompt: str,
        user_text: str,
        *,
        history: Sequence[ChatMessage] = (),
        max_tokens: int = MAX_REPLY_TOKENS,
    ) -> Iterator[str]:
        if self._model is None:
            raise RuntimeError("model not loaded -- call load() at startup, never per request")
        from mlx_lm import stream_generate

        for chunk in stream_generate(
            self._model,
            self._tok,
            self._render(system_prompt, history, user_text),
            max_tokens=max_tokens,
        ):
            yield chunk.text

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        *,
        history: Sequence[ChatMessage] = (),
        max_tokens: int = MAX_REPLY_TOKENS,
    ) -> str:
        return "".join(
            self.stream(system_prompt, user_text, history=history, max_tokens=max_tokens)
        ).strip()


class StubLlm:
    """Deterministic stand-in for tests.

    Exists because the real model is 14.2 GB. It implements the same Protocol, so
    anything type-checking against LlmService works with either -- but it is never
    reachable from the app: main.py constructs MlxLlm, and a stub silently
    substituted in production would be exactly the "silent fallback" T1.7 forbids.
    """

    def __init__(self, reply: str = "いいですね。何を食べましたか。") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []
        self.histories: list[tuple[ChatMessage, ...]] = []

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        *,
        history: Sequence[ChatMessage] = (),
        max_tokens: int = 64,
    ) -> str:
        self.calls.append((system_prompt, user_text))
        self.histories.append(tuple(history))
        return self.reply

    def stream(
        self,
        system_prompt: str,
        user_text: str,
        *,
        history: Sequence[ChatMessage] = (),
        max_tokens: int = 64,
    ) -> Iterator[str]:
        self.calls.append((system_prompt, user_text))
        self.histories.append(tuple(history))
        yield from self.reply

    def status(self) -> LlmStatus:
        return LlmStatus(model="stub", loaded=True, resident_gb=0.0)
