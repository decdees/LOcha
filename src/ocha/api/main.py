"""FastAPI application.

Phase 1 exposes POST /turn and nothing else -- there is no client (PRD §10).
/health exists for liveness and, from T1.7, for reporting model residency.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Ocha", docs_url=None, redoc_url=None)


class Health(BaseModel):
    status: str
    # T1.7 adds: model_loaded, resident_memory_gb


@app.get("/health")
def health() -> Health:
    return Health(status="ok")
