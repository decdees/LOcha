"""Starter vocabulary.

50 beginner items skewed to travel and conversational survival, per PRD §2.
Every item enters with a fresh py-fsrs Card, so nothing is pre-marked as known --
the scheduler learns the pool from real reviews.
"""

from __future__ import annotations

import sqlite3

from fsrs import Card

# (content, reading, meaning_en)
VOCAB: list[tuple[str, str, str]] = [
    ("私", "わたし", "I, me"),
    ("これ", "これ", "this"),
    ("それ", "それ", "that"),
    ("あれ", "あれ", "that over there"),
    ("ここ", "ここ", "here"),
    ("どこ", "どこ", "where"),
    ("何", "なに", "what"),
    ("誰", "だれ", "who"),
    ("いつ", "いつ", "when"),
    ("いくら", "いくら", "how much"),
    ("今日", "きょう", "today"),
    ("明日", "あした", "tomorrow"),
    ("昨日", "きのう", "yesterday"),
    ("朝", "あさ", "morning"),
    ("夜", "よる", "night"),
    ("時間", "じかん", "time, hour"),
    ("水", "みず", "water"),
    ("お茶", "おちゃ", "tea"),
    ("コーヒー", "コーヒー", "coffee"),
    ("ご飯", "ごはん", "rice, meal"),
    ("肉", "にく", "meat"),
    ("魚", "さかな", "fish"),
    ("野菜", "やさい", "vegetables"),
    ("店", "みせ", "shop"),
    ("駅", "えき", "station"),
    ("電車", "でんしゃ", "train"),
    ("家", "いえ", "house, home"),
    ("トイレ", "トイレ", "toilet"),
    ("ホテル", "ホテル", "hotel"),
    ("空港", "くうこう", "airport"),
    ("切符", "きっぷ", "ticket"),
    ("お金", "おかね", "money"),
    ("カード", "カード", "card"),
    ("友達", "ともだち", "friend"),
    ("先生", "せんせい", "teacher"),
    ("本", "ほん", "book"),
    ("食べる", "たべる", "to eat"),
    ("飲む", "のむ", "to drink"),
    ("行く", "いく", "to go"),
    ("来る", "くる", "to come"),
    ("見る", "みる", "to see, watch"),
    ("する", "する", "to do"),
    ("ある", "ある", "to exist (inanimate)"),
    ("いる", "いる", "to exist (animate)"),
    ("わかる", "わかる", "to understand"),
    ("好き", "すき", "liked, favourite"),
    ("大きい", "おおきい", "big"),
    ("小さい", "ちいさい", "small"),
    ("高い", "たかい", "expensive, tall"),
    ("安い", "やすい", "cheap"),
]


def seed(conn: sqlite3.Connection) -> int:
    """Insert starter items. Idempotent -- existing content is left alone."""
    rows = []
    for content, reading, meaning in VOCAB:
        card = Card()
        rows.append(
            (
                "vocab",
                content,
                reading,
                meaning,
                card.to_json(),
                card.due.isoformat(),
                card.stability or 0.0,
            )
        )
    before: int = conn.execute("SELECT count(*) FROM items").fetchone()[0]
    conn.executemany(
        "INSERT OR IGNORE INTO items"
        " (kind, content, reading, meaning_en, fsrs_json, due, stability)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    after: int = conn.execute("SELECT count(*) FROM items").fetchone()[0]
    return after - before
