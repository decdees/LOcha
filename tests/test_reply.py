from __future__ import annotations

import pytest

from ocha.tutor.reply import TutorReplyError, parse_tutor_reply


def test_structured_reply_has_learner_aids() -> None:
    reply = parse_tutor_reply(
        '{"japanese":"こんにちは。お茶をください。","english":"Hello. Tea, please."}'
    )
    assert reply.japanese == "こんにちは。お茶をください。"
    assert reply.romaji.lower() == "konnichiwa. ocha o kudasai."
    assert reply.meaning_en == "Hello. Tea, please."


@pytest.mark.parametrize(
    "raw",
    [
        "こんにちは",
        '{"japanese":"こんにちは"}',
        '{"japanese":"こんにちは","english":""}',
        '{"japanese":"こんにちは","english":"Hello","extra":"no"}',
    ],
)
def test_malformed_structured_reply_fails_closed(raw: str) -> None:
    with pytest.raises(TutorReplyError):
        parse_tutor_reply(raw)
