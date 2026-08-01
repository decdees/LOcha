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
import pathlib
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ocha.db import connect, migrate
from ocha.db.seed import seed
from ocha.scheduling import ItemScheduler
from ocha.speech.wire import SAMPLE_RATE
from ocha.tutor.llm import LlmService, MlxLlm
from ocha.tutor.observation import Evidence
from ocha.tutor.turn import run_turn

# Set to skip the model load -- for `make dev` when you only want the HTTP surface,
# and for the test suite, which cannot hold a 14.2 GB model. /health reports
# loaded=false, so a skipped load is never mistaken for a working one.
SKIP_MODEL = os.environ.get("OCHA_SKIP_MODEL") == "1"

WEB_DIR = pathlib.Path(__file__).resolve().parents[3] / "web"

# One shared connection, one writer.
#
# One lock serializes the single user's turn mutation. MLX work, including status,
# is submitted to the dedicated inference worker that loaded both models; FastAPI's
# event loop and threadpool never touch MLX directly.
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
    observations: dict[int, Evidence] = {}
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
    app.state.worker = None
    if not SKIP_MODEL:
        from ocha.inference import InferenceWorker, WorkerLlm
        from ocha.speech.asr import OchaWhisper
        from ocha.speech.filler import FillerBank
        from ocha.speech.repair import synthesise_repair
        from ocha.speech.tts import VoicevoxTTS

        # ONE thread owns both models for the process lifetime. Loading happens on
        # that thread, inside start(), because MLX GPU streams are thread-local to
        # whoever ran load() -- standing constraint 6. T2.6 measured why it is a
        # thread and not the event loop: inference there blocks frame delivery for
        # its whole duration. See benchmarks/voice-loop.md.
        worker = InferenceWorker()
        asr = OchaWhisper(sample_rate=SAMPLE_RATE, worker=worker)
        worker.start(llm.load, asr.warm)  # type: ignore[attr-defined]
        app.state.worker = worker
        app.state.asr = asr
        llm = WorkerLlm(worker, llm)

        # Pre-synthesised filled pauses. ~0.4 s each here, 0 ms in the turn.
        voice = VoicevoxTTS(sample_rate=SAMPLE_RATE)
        app.state.fillers = await FillerBank.synthesise(voice, SAMPLE_RATE)
        app.state.repair_audio = await synthesise_repair(voice)
    app.state.llm = llm
    yield
    if app.state.worker is not None:
        app.state.worker.stop()


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
        observations=result.observations,
        ratings=result.ratings,
        usage=result.usage,
    )


@app.websocket("/ws")
async def ws(websocket: WebSocket, loopback: bool = False) -> None:
    """The voice loop (T2.1-T2.5). One client, one connection, its whole life.

    `async def` for the same reason /turn is: the pipeline runs MLX inference
    (whisper, then the LLM) and MLX GPU streams are thread-local to whichever
    thread ran `load()`. Standing constraint 6.

    `?loopback=1` runs the diagnostic echo pipeline instead -- no VAD, no models.
    That is the by-ear audio check for HFP headset capture (ARCHITECTURE risk 4).
    """
    await websocket.accept()
    from ocha.speech.pipeline import run_session

    probe = await run_session(
        websocket,
        app.state.conn,
        app.state.scheduler,
        app.state.grammar,
        app.state.llm,
        loopback=loopback,
        # The warmed instance, not a fresh one -- a second OchaWhisper would pay
        # the load cost again inside the first turn.
        asr=getattr(app.state, "asr", None),
        tts=getattr(app.state, "tts", None),
        fillers=getattr(app.state, "fillers", None),
        repair_audio=getattr(app.state, "repair_audio", None),
    )
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


# LAST, after every route. A mount at "/" matches everything, and Starlette checks
# routes in registration order -- mounted earlier it silently shadows /health and
# /turn. `html=True` serves web/index.html at "/".
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
