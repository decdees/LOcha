"""Validated learner-facing fields from one quarantined model response."""

from __future__ import annotations

import json
from dataclasses import dataclass

import cutlet

_ROMAJI = cutlet.Cutlet("hepburn")
_ROMAJI.use_foreign_spelling = False
_ROMAJI.update_mapping("を", "o")
_ROMAJI.add_exception("こんにちは", "konnichiwa")
_ROMAJI.add_exception("こんばんは", "konbanwa")


class TutorReplyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TutorReply:
    japanese: str
    romaji: str
    meaning_en: str


def parse_tutor_reply(raw: str) -> TutorReply:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TutorReplyError("the tutor returned an invalid response; please try again") from exc
    if not isinstance(payload, dict) or set(payload) != {"japanese", "english"}:
        raise TutorReplyError("the tutor returned an invalid response; please try again")
    japanese = payload["japanese"]
    english = payload["english"]
    if not isinstance(japanese, str) or not japanese.strip():
        raise TutorReplyError("the tutor returned no Japanese text; please try again")
    if not isinstance(english, str) or not english.strip():
        raise TutorReplyError("the tutor returned no English meaning; please try again")
    try:
        romaji = str(_ROMAJI.romaji(japanese)).strip()
    except Exception as exc:
        raise TutorReplyError("the tutor reading could not be generated; please try again") from exc
    if not romaji:
        raise TutorReplyError("the tutor reading could not be generated; please try again")
    return TutorReply(japanese.strip(), romaji, english.strip())
