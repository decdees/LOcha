"""T0.2 — record the 20-utterance voice corpus.

Reads sentences from transcripts.json, records one 16 kHz mono WAV each, and
refuses to advance on a take that is too quiet to be speech.

Two traps this exists to avoid:

1. The default macOS audio input is often NOT the built-in mic. On this machine
   `ffmpeg -f avfoundation -list_devices` reports an Aggregate Device at index 0
   and BlackHole at index 1. Recording the default yields 20 silent files, and
   you discover it as a ~100% CER row that looks like a model finding. So the
   device index is pinned explicitly and every take is level-checked.

2. Ground truth must be what you ACTUALLY said, not what was printed. CER against
   a sentence you misread measures nothing. After each take you confirm the text,
   and a correction is written straight back to transcripts.json.

Usage:
    uv run python benchmarks/corpus/record.py --self-check   # test the guard, no mic
    uv run python benchmarks/corpus/record.py --list         # show audio devices
    uv run python benchmarks/corpus/record.py --test-mic     # 2s level check
    uv run python benchmarks/corpus/record.py                # record the corpus
"""

from __future__ import annotations

import argparse
import array
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import wave

HERE = pathlib.Path(__file__).parent
TRANSCRIPTS = HERE / "transcripts.json"

RATE = 16_000
PREFERRED_DEVICE = "MacBook Pro Microphone"

# The guard answers "is there speech here", which is not the same question as
# "is this loud". A global RMS threshold conflates the two: raise it enough to
# reject a lone click on a dead track and it starts rejecting quiet speech.
# So presence is measured as the fraction of 20 ms frames carrying energy --
# a crude VAD. A click is one active frame in a hundred; speech is most of them.
FRAME_MS = 20
# 0.004, not 0.01. The first corpus came back peaking at 0.07-0.13 (about -22
# dBFS), and at a 0.01 frame floor that under-counted voiced frames by ~2x --
# which made every take look like it held less speech than it did, and would
# have hidden a genuinely truncated recording behind an already-low number.
FRAME_ACTIVE_RMS = 0.004
MIN_ACTIVE_FRACTION = 0.10  # speech with pauses still clears this comfortably
MIN_PEAK = 0.02
MIN_SECONDS = 0.5

# Japanese runs ~7 morae/sec conversationally. Used only to warn, never to
# reject: this catches "you read something other than the target line", which
# is the failure that silently invalidated the first corpus.
MORAE_PER_SEC = 4.5  # beginner pace, not the ~7 of a native speaker


# --------------------------------------------------------------------------
# level check -- the guard
# --------------------------------------------------------------------------


def levels(path: pathlib.Path) -> tuple[float, float, float]:
    """Return (peak, active_fraction, seconds). peak normalized to 0..1."""
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        rate = w.getframerate()
        n = w.getnframes()
        seconds = n / rate
        samples = array.array("h", w.readframes(n))
    if not samples:
        return 0.0, 0.0, seconds

    peak = max(abs(s) for s in samples) / 32768
    size = int(rate * FRAME_MS / 1000)
    frames = [samples[i : i + size] for i in range(0, len(samples) - size + 1, size)]
    if not frames:
        return peak, 0.0, seconds
    active = sum(
        1
        for f in frames
        if (sum(s * s for s in f) / len(f)) ** 0.5 / 32768 >= FRAME_ACTIVE_RMS
    )
    return peak, active / len(frames), seconds


def verdict(path: pathlib.Path) -> tuple[bool, str]:
    peak, active, seconds = levels(path)
    stats = f"peak {peak:.3f}, active {active:.0%}, {seconds:.1f}s"
    if seconds < MIN_SECONDS:
        return False, f"too short ({stats})"
    if peak < MIN_PEAK:
        return False, f"no signal -- wrong input device or muted mic? ({stats})"
    if active < MIN_ACTIVE_FRACTION:
        return False, f"signal present but almost no speech -- a pop, not a take? ({stats})"
    return True, stats


# --------------------------------------------------------------------------
# devices
# --------------------------------------------------------------------------


def list_devices() -> list[tuple[int, str]]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    ).stderr
    audio = out.split("audio devices:")[-1]
    return [(int(m[1]), m[2].strip()) for m in re.finditer(r"\[(\d+)\] (.+)", audio)]


def pick_device(explicit: int | None) -> int:
    devices = list_devices()
    if explicit is not None:
        if explicit not in [i for i, _ in devices]:
            sys.exit(f"device {explicit} not found. Available:\n" + _fmt(devices))
        return explicit
    for i, name in devices:
        if PREFERRED_DEVICE.lower() in name.lower():
            return i
    sys.exit(
        f"could not find {PREFERRED_DEVICE!r}. Pass --device N explicitly.\n"
        "Do NOT use an Aggregate Device or BlackHole -- they record silence.\n" + _fmt(devices)
    )


def _fmt(devices: list[tuple[int, str]]) -> str:
    return "\n".join(f"  [{i}] {n}" for i, n in devices)


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------


def record(path: pathlib.Path, device: int) -> None:
    proc = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "avfoundation", "-i", f":{device}",
            "-ar", str(RATE), "-ac", "1", "-c:a", "pcm_s16le",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
    )
    try:
        input()
    finally:
        assert proc.stdin is not None
        proc.stdin.write(b"q")  # graceful stop, so ffmpeg finalizes the WAV header
        proc.stdin.flush()
        proc.wait(timeout=10)


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------


def load() -> dict:
    return json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))


def save(data: dict) -> None:
    TRANSCRIPTS.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    name = dict(list_devices())[device]
    data = load()
    items = data["utterances"]

    print(f"\ninput device [{device}] {name}   {RATE} Hz mono 16-bit\n")
    print("Enter to start a take, Enter again to stop. Ctrl-C to quit.\n")

    for item in items:
        wav = HERE / f"{item['id']}.wav"
        if wav.exists() and not args.force:
            ok, _ = verdict(wav)
            if ok and item.get("verified"):
                print(f"  {item['id']}  done, skipping")
                continue

        while True:
            print(f"\n── {item['id']} ({item['subset']})")
            # Romaji is the line to read. The English gloss is NEVER shown here:
            # v1 printed it under the Japanese and the whole corpus came back in
            # English, because a beginner reads the line they can actually read.
            # The Japanese script is reference only -- it is the CER target.
            print(f"   SAY:     {item['romaji']}")
            print(f"   script:  {item['text']}")
            input("   Enter to start... ")
            record(wav, device)
            ok, stats = verdict(wav)
            if not ok:
                print(f"   ✗ REJECTED: {stats}")
                if input("   retry? [Y/n] ").strip().lower() == "n":
                    sys.exit("aborted -- corpus incomplete")
                continue
            print(f"   ✓ {stats}")

            expected = expected_seconds(item["romaji"])
            _, active, sec = levels(wav)
            ratio = (sec * active) / expected
            if not 0.5 <= ratio <= 2.5:
                longer = ratio > 2.5
                print(
                    f"   ⚠ voiced speech is {ratio:.1f}x the expected length for this line."
                    + (
                        "\n     Did you read the script line as well, or say something extra?"
                        if longer
                        else "\n     Did the recording start late or cut off early?"
                    )
                )

            # Ground truth must match what was actually said.
            reply = input("   Read exactly as written? [Y/n/r=redo] ").strip().lower()
            if reply == "r":
                continue
            if reply == "n":
                actual = input("   Type what you actually said: ").strip()
                if actual:
                    item["text"] = actual
                    print("   transcript corrected")
            item["verified"] = True
            save(data)
            break

    done = sum(1 for i in items if i.get("verified"))
    print(f"\n{done}/{len(items)} verified. transcripts.json updated.")
    if done == len(items):
        print("T0.2 complete -- T0.3 can run.")


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def _write_wav(path: pathlib.Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(array.array("h", samples).tobytes())


def self_check() -> None:
    """Test the guard against synthesized audio. No microphone needed.

    The guard is the measurement instrument here: if it wrongly passes silence,
    the whole corpus can be empty and T0.3 reports it as a model failure.
    """
    import math

    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        n = RATE * 2

        silent = tmp / "silent.wav"
        _write_wav(silent, [0] * n)
        ok, why = verdict(silent)
        assert not ok, f"guard PASSED pure silence: {why}"
        print(f"  silence      rejected  ({why})")

        # A single loud click on an otherwise dead track: peak is near full scale,
        # so peak alone would pass it. This is the case that forced frame-based
        # activity instead of a global RMS threshold.
        click = tmp / "click.wav"
        s = [0] * n
        s[n // 2] = 30000
        _write_wav(click, s)
        ok, why = verdict(click)
        assert not ok, f"guard PASSED a lone click: {why}"
        print(f"  lone click   rejected  ({why})")

        speech = tmp / "tone.wav"
        _write_wav(speech, [int(12000 * math.sin(i * 0.05)) for i in range(n)])
        ok, why = verdict(speech)
        assert ok, f"guard REJECTED a clear signal: {why}"
        print(f"  clear tone   accepted  ({why})")

        # Quiet but real speech must still pass -- the guard must not become a
        # loudness gate. ~ -32 dBFS, well below a comfortable recording level.
        quiet = tmp / "quiet.wav"
        _write_wav(quiet, [int(800 * math.sin(i * 0.05)) for i in range(n)])
        ok, why = verdict(quiet)
        assert ok, f"guard REJECTED quiet speech: {why}"
        print(f"  quiet speech accepted  ({why})")

        # Speech with long pauses, as in a real take: bursts separated by silence.
        gappy = tmp / "gappy.wav"
        s = []
        for block in range(10):
            loud = block % 3 != 0  # ~2/3 active
            s += [int((10000 if loud else 0) * math.sin(i * 0.05)) for i in range(n // 10)]
        _write_wav(gappy, s)
        ok, why = verdict(gappy)
        assert ok, f"guard REJECTED speech with pauses: {why}"
        print(f"  with pauses  accepted  ({why})")

        short = tmp / "short.wav"
        _write_wav(short, [int(12000 * math.sin(i * 0.05)) for i in range(RATE // 10)])
        ok, why = verdict(short)
        assert not ok, f"guard PASSED a 0.1s take: {why}"
        print(f"  0.1s take    rejected  ({why})")

    print("\nself-check passed.")


def expected_seconds(romaji: str) -> float:
    """Rough spoken length. Japanese morae ~= vowel count; English ~3 chars/syllable."""
    ja = sum(1 for c in romaji.lower() if c in "aeiou")
    return max(0.6, ja / MORAE_PER_SEC)


def verify_corpus() -> None:
    """Confirm the recordings are in the language we think they are.

    This exists because the first corpus was recorded almost entirely in English
    -- record.py v1 displayed the English gloss under the Japanese, and a
    beginner reads the line they can read. It cost a full three-model bake-off
    to discover, from a CER table reading 95-130%. Sixty seconds here would have
    caught it.
    """
    import torch
    from transformers import pipeline

    data = load()
    p = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        dtype=torch.float16,
        device="mps",
    )
    # transformers 5.x: the model's own forced_decoder_ids silently overrides the
    # language/task kwargs. Without this line, forcing "ja" is a no-op.
    p.model.generation_config.forced_decoder_ids = None

    def jp_ratio(s: str) -> float:
        ja = sum(1 for c in s if "぀" <= c <= "ヿ" or "一" <= c <= "鿿")
        latin = sum(1 for c in s if c.isascii() and c.isalpha())
        return ja / max(1, ja + latin)

    import difflib

    def close(a: str, b: str) -> float:
        norm = lambda s: "".join(s.lower().split())
        return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()

    bad = []
    for u in data["utterances"]:
        wav = HERE / f"{u['id']}.wav"
        if not wav.exists():
            continue
        hyp = p(str(wav), generate_kwargs={"language": "ja", "task": "transcribe"})["text"].strip()
        want, got = jp_ratio(u["text"]), jp_ratio(hyp)

        # (a) reference is Japanese but the output is not.
        #     Exempt takes that match the romaji: whisper sometimes renders correct
        #     Japanese in Latin script ("This is Oshii Desu-ne", "TOKYO STATION MADE
        #     IKURU DESUKA"). That is a script choice, not a wrong-language recording,
        #     and flagging it sends the speaker back to re-record perfectly good audio.
        sounds_right = close(hyp, u.get("romaji", "")) > 0.55
        wrong_script = want > 0.5 and got < 0.3 and not sounds_right
        # (b) the output resembles the English gloss more than the target line.
        #     Catches gloss-reading even when the target is itself English-heavy,
        #     which (a) cannot see -- e.g. "I am an engineer" vs "watashi wa
        #     engineer desu". Both are Latin script; only this test separates them.
        gloss = u.get("gloss", "")
        read_gloss = (
            bool(gloss)
            and not sounds_right  # same exemption: matching the romaji means it was said right
            and close(hyp, gloss) > close(hyp, u["text"]) + 0.15
        )

        flag = wrong_script or read_gloss
        why = "script" if wrong_script else ("gloss" if read_gloss else "")
        print(
            f"  {u['id']} {'FLAG ' + why if flag else 'ok       '}"
            f" ref[{want:.0%} ja] hyp[{got:.0%} ja]  {hyp[:48]}"
        )
        if flag:
            bad.append(u["id"])

    print(f"\n{len(bad)} of {len(data['utterances'])} recordings look like the wrong language: {bad or 'none'}")
    if bad:
        print("Re-record those ids. Read the SAY line (romaji) -- never the script or any English.")


def test_mic(device: int | None) -> None:
    d = pick_device(device)
    print(f"[{d}] {dict(list_devices())[d]}")
    print("Say something, then press Enter.")
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "t.wav"
        input("Enter to start... ")
        record(p, d)
        ok, stats = verdict(p)
        print(("✓ " if ok else "✗ ") + stats)
        if not ok:
            print("Wrong device, or the mic is muted. Try --list and --device N.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, help="avfoundation audio index; default: built-in mic")
    ap.add_argument("--force", action="store_true", help="re-record takes already done")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--test-mic", action="store_true")
    ap.add_argument("--verify", action="store_true", help="check recordings are actually Japanese")
    a = ap.parse_args()

    if a.list:
        print(_fmt(list_devices()))
    elif a.self_check:
        self_check()
    elif a.verify:
        verify_corpus()
    elif a.test_mic:
        test_mic(a.device)
    else:
        main(a)
