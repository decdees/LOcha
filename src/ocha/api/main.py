"""FastAPI application.

Phase 1 exposes POST /turn and nothing else -- there is no client (PRD §10).

The model is loaded in the lifespan startup, once, and kept warm for the process
lifetime. Loading lazily on first request would move a measured 6-7 s cold load
into a request path, which project standing constraint 3 classifies as a correctness bug
rather than a slow path.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel, Field

from ocha.db import connect, migrate
from ocha.db.seed import seed
from ocha.scheduling import ItemScheduler
from ocha.tutor.llm import LlmService, MlxLlm
from ocha.tutor.turn import run_turn

# Set to skip the model load -- for `make dev` when you only want the HTTP surface,
# and for the test suite, which cannot hold a 14.2 GB model. /health reports
# loaded=false, so a skipped load is never mistaken for a working one.
SKIP_MODEL = os.environ.get("OCHA_SKIP_MODEL") == "1"

# One shared connection, one writer.
#
# /turn is `async def` deliberately. MLX GPU streams are THREAD-LOCAL: the model is
# loaded on the event-loop thread during lifespan startup, and generating from a
# threadpool worker fails with "There is no Stream(gpu, 1) in current thread".
# FastAPI dispatches `def` endpoints to a threadpool and `async def` endpoints on
# the event loop, so async keeps inference on the thread that owns the stream.
#
# ponytail: this blocks the event loop for the 1-3 s a generation takes. Acceptable
# because there is exactly one user (PRD §4 non-goals) and nothing else needs
# serving concurrently. If that ever changes, the upgrade is a dedicated
# single-threaded inference worker with a queue -- not a threadpool, which would
# reintroduce this exact bug.
_turn_lock = threading.Lock()


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    session_id: int | None = None


class GrammarPayload(BaseModel):
    """Mirrors GrammarResponse. Like it, has NO field for model text (FR-5)."""

    kind: str
    text: str
    entry_id: str | None = None
    examples: list[str] = []
    hindi_contrast: str | None = None
    interference_warning: bool = False


class TurnResponse(BaseModel):
    session_id: int
    turn_id: int
    reply: str | None = None
    grammar: GrammarPayload | None = None
    targets: list[str] = []
    ratings: dict[int, int] = {}
    usage: dict[int, str] = {}


class Health(BaseModel):
    status: str
    model: str
    model_loaded: bool
    resident_memory_gb: float
    grammar_entries: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from ocha.tutor.grammar import load_grammar

    # Fail fast: a malformed reference means the firewall has nothing to serve,
    # and discovering that mid-conversation is worse than not starting.
    app.state.grammar = load_grammar()

    conn = connect()
    migrate(conn)
    seed(conn)  # idempotent
    app.state.conn = conn
    app.state.scheduler = ItemScheduler(conn)

    llm: LlmService = MlxLlm()
    if not SKIP_MODEL:
        # Raises if unavailable. There is no cloud fallback to fall back to.
        llm.load()  # type: ignore[attr-defined]
    app.state.llm = llm
    yield


app = FastAPI(title="Ocha", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.post("/turn")
async def turn(req: TurnRequest) -> TurnResponse:
    """The only Phase 1 endpoint. No client exists (PRD §10)."""
    try:
        with _turn_lock:
            result = run_turn(
                app.state.conn,
                app.state.scheduler,
                app.state.grammar,
                app.state.llm,
                req.text,
                session_id=req.session_id,
            )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    grammar = None
    if result.grammar is not None:
        g = result.grammar
        grammar = GrammarPayload(
            kind=g.kind,
            text=g.text,
            entry_id=g.entry_id,
            examples=list(g.examples),
            hindi_contrast=g.hindi_contrast,
            interference_warning=g.interference_warning,
        )
    return TurnResponse(
        session_id=result.session_id,
        turn_id=result.turn_id,
        reply=result.reply,
        grammar=grammar,
        targets=list(result.targets),
        ratings=result.ratings,
        usage=result.usage,
    )


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """The voice loop (T2.1). One client, one connection, its whole life.

    `async def` for the same reason /turn is: the pipeline will run MLX inference
    (whisper, then the LLM) and MLX GPU streams are thread-local to whichever
    thread ran `load()`. Standing constraint 6.
    """
    await websocket.accept()
    from ocha.speech.pipeline import run_session

    probe = await run_session(websocket)
    logging.getLogger(__name__).info("ws session ended: %s", probe.report())


@app.get("/health")
def health() -> Health:
    llm: LlmService = app.state.llm
    st = llm.status()
    return Health(
        status="ok" if st.loaded else "degraded",
        model=st.model,
        model_loaded=st.loaded,
        resident_memory_gb=st.resident_gb,
        grammar_entries=len(app.state.grammar),
    )
