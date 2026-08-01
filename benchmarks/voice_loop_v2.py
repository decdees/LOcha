"""Validate and score a version-2 iPhone voice capture without post-selection.

The input is the PWA/client-clock capture. This evaluator never overwrites an
existing result and never compares client timestamps to server wall clocks.

Usage: uv run python benchmarks/voice_loop_v2.py capture.json
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocha.measurement import (
    AudioInterval,
    ExchangeCapture,
    InstrumentFailure,
    VisibleChange,
    gate_summary,
    measure_exchange,
)
from ocha.speech.wire import AudioKind

HERE = Path(__file__).parent


def _capture(raw: dict[str, Any]) -> ExchangeCapture:
    exchange_id = uuid.UUID(str(raw["exchange_id"]))
    return ExchangeCapture(
        exchange_id=exchange_id,
        speech_end_ms=float(raw["speech_end_ms"]),
        visible=tuple(
            VisibleChange(exchange_id, int(event["seq"]), float(event["at_ms"]))
            for event in raw.get("visible", [])
        ),
        audio=tuple(
            AudioInterval(
                exchange_id=uuid.UUID(str(clip["exchange_id"])),
                sequence=int(clip["seq"]),
                kind=AudioKind(int(clip["kind"])),
                start_ms=float(clip["start_ms"]),
                duration_ms=float(clip["duration_ms"]),
                cancelled_at_ms=(
                    float(clip["cancelled_at_ms"])
                    if clip.get("cancelled_at_ms") is not None
                    else None
                ),
            )
            for clip in raw.get("audio", [])
        ),
        asr=str(raw["asr"]),  # type: ignore[arg-type]
    )


def evaluate(source: Path) -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != 2:
        raise InstrumentFailure("capture is not protocol version 2")
    captures = [_capture(row) for row in payload.get("turns", [])]
    rows: list[dict[str, object]] = []
    for capture in captures:
        try:
            metric = measure_exchange(capture)
            rows.append(
                {
                    "exchange_id": str(metric.exchange_id),
                    "asr": metric.asr,
                    "g1b_ms": metric.g1b_ms,
                    "longest_uncovered_ms": metric.longest_uncovered_ms,
                    "instrument_failure": None,
                }
            )
        except InstrumentFailure as exc:
            rows.append(
                {
                    "exchange_id": str(capture.exchange_id),
                    "asr": capture.asr,
                    "g1b_ms": None,
                    "longest_uncovered_ms": None,
                    "instrument_failure": str(exc),
                }
            )
    return {
        "protocol_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "capture_path": payload.get("capture_path"),
        "gate": gate_summary(captures),
        "turns": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or HERE / f"voice-loop-v2-{stamp}.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite historical result: {out}")
    result = evaluate(args.capture)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["gate"], indent=2))
    print(out)


if __name__ == "__main__":
    main()
