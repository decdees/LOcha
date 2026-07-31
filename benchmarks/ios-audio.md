# T2.1 pre-flight — iOS audio behaviour

**Date:** 31 July 2026 · iPhone, iOS 18.7, AirPods connected · WebKit 605.1.15
**Served over HTTPS with a self-signed certificate** (Tailscale not yet set up). Safari labels this "Not Secure", but the page confirmed `isSecureContext: true`, so `getUserMedia` behaved as it will in production.

---

## (b) Output routing — **PASS**. The structural risk did not reproduce.

iOS Safari is documented to force audio output to the built-in speaker when `getUserMedia` starts, with WebKit bugs still open. **It did not happen here.** A 440 Hz tone played with the microphone open came out of the **AirPods**.

This was the finding that could have ended the current design: forced speaker output would have pinned playback to the phone permanently, made private practice impossible, and fed the tutor's voice back into the open mic during barge-in. **Continuous listening (FR-1) is viable as designed.** No client redesign needed, no push-to-talk fallback required.

## (a) `getUserMedia` in standalone mode — **PASS** (third run)

Third run, opened from the **home-screen icon**: `Standalone (home screen): true`, `Secure context: true`, and `getUserMedia` returned a live track (`iPhone Microphone`). The installed-PWA restriction that Safari applied in earlier iOS versions does not apply here.

Both pre-flight gates therefore pass. **The PWA client is viable exactly as PRD FR-9 described it** — installable to the home screen, continuous listening, output on the headset.

### This result is now the fallback, not the plan

After it passed, the client was changed to a **native SwiftUI app** (see PRD FR-9, ARCHITECTURE §2). That is a deliberate upgrade, not a rescue: native gives explicit `AVAudioSession` category and route control, background audio, and no dependence on WebKit's evolving media policy. The PWA remains fully documented and proven, and this file is why — if the native path stalls on signing or schedule, the fallback is known to work rather than hoped to.

**What going native does NOT fix:** iOS cannot pair the built-in microphone with A2DP output. Calling `setPreferredInput(builtInMic)` forces output to the speaker; using the headset mic means HFP for both directions. The variable-input-device finding below survives the pivot unchanged.

---

## New finding: the input device is NOT stable between runs

Three runs, all with AirPods connected:

| run | mode | `track.label` | `AudioContext.sampleRate` |
|---|---|---|---|
| 1 | Safari tab | **AirPods** | 24000 |
| 2 | Safari tab | **iPhone Microphone** | 24000 |
| 3 | standalone | **iPhone Microphone** | 24000 |

**The microphone iOS hands the page is not predictable.** Same phone, same connected headset, same page — different capture device. The app cannot assume either, and the two have materially different acoustic characteristics: a Bluetooth duplex headset mic versus the phone's own array.

**Correction to what this report said earlier.** It stated the microphone "is the AirPods at 24 kHz". Two things were wrong with that. The device is not fixed, as above. And **24000 is the `AudioContext` sample rate, not the microphone's** — that figure comes from `ctx.sampleRate`, logged separately from the track settings. It does indicate iOS dropped the audio session to a lower rate once the mic went live, which is consistent with duplex operation, but it is not a measurement of the capture device. The track's own `sampleRate` is in the truncated settings JSON and has not been read.

**Why this is a problem for the ASR numbers.** The T0.2 corpus was recorded on the **MacBook Pro's built-in microphone**, and `asr.md` already flags that whisper-large-v3's 2.56% CER was measured on rehearsed read-aloud speech in a quiet room. This adds a second, independent domain shift — and a worse one than a single alternative path would be, because the production input is **variable**. The corpus characterises one capture device; production may use either of two, neither of them the one measured.

`2.56` was already marked as not load-bearing. This is a concrete reason why, not a hypothetical one.

**Recorded as a Phase 2 task rather than fixed now** — see T2.3. The fix is to re-record part of the corpus through the actual production path (AirPods → iPhone → Tailscale → ASR) and re-measure. Doing it before the transport exists would just be guessing at the pipeline.

---

## Open

- **AirPods-path CER** — unmeasured, folded into T2.3. Now measured through the native app rather than the PWA, but the question is identical: the corpus characterises the MacBook's built-in mic, production uses one of two phone-side devices.
- **Tailscale** — installed (standalone build, 1.98.10) but never signed in: no daemon, no system extension, CLI unusable. Needed before T2.7 puts anything on the phone.

## Reproduce

```bash
python3 web/serve_https.py <cert-dir>   # prints the LAN URL
```
