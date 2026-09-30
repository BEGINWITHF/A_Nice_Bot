#!/usr/bin/env python3
"""
A dumb sense organ - the device half of the capability part (STEP-2).

CAP-7  - the device is a sense, not a brain: it opens an eye or an ear and
         ships what they produced, raw.  No salience, no judgement about what
         is worth sending, and no analysis - decoding and perception happen
         on the host (author's answer 2026-09-30: "画面被实时传回服务器，
         而服务器再让 ai 感知").
CAP-8  - the host address is configured by hand in configs/uplink.json.
OPEN-7 - the bearer token comes from the same file; industry standard, as the
         author asked ("你按照行业标准就好").

This file deliberately imports nothing from `core/`.  It needs the standard
library plus opencv (for an eye) and pyaudio (for an ear), and that is all:
there is no way to run the brain on a device by accident.

    copy  device.py, uplink_config.py, configs/uplink.json
    run   python device.py
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

import uplink_config

JPEG_QUALITY = 80
AUDIO_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_SECONDS = 1.0
TIMEOUT_S = 10.0


class PushFailed(Exception):
    """Could not reach the host, or the host refused the observation."""


def base_url(cfg):
    return "http://%s:%s" % (cfg["host"], cfg["port"])


def headers_for(cfg, content_type, extra=None):
    out = {
        "Content-Type": content_type,
        "Authorization": "Bearer %s" % cfg["token"],
        "X-Device-Id": cfg["device_id"],
        # The host cares when the sample was taken, not when it arrived:
        # the pipeline windows observations on a clock (OPEN-16), so a frame
        # that queued for a second must not be recorded a second late.
        "X-Captured-At": "%.3f" % time.time(),
    }
    for key, value in (extra or {}).items():
        out[key] = str(value)
    return out


def push(cfg, content_type, body, extra=None):
    """POST one raw observation.  Raises PushFailed with a usable message."""
    request = urllib.request.Request(
        base_url(cfg) + "/v1/observations",
        data=body,
        headers=headers_for(cfg, content_type, extra),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            response.read()
            return response.status
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8", "replace")).get("error", "")
        except ValueError:
            pass
        if exc.code == 401:
            raise PushFailed(
                "host rejected the token (401). The two copies of "
                "configs/uplink.json must share one `token`.") from None
        raise PushFailed("host answered %s%s" % (
            exc.code, ": %s" % detail if detail else "")) from None
    except urllib.error.URLError as exc:
        raise PushFailed(
            "cannot reach %s (%s). CAP-8 says `host` is typed by hand - "
            "check it is this machine's address and that the port matches."
            % (base_url(cfg), getattr(exc, "reason", exc))) from None
    except OSError as exc:
        raise PushFailed("cannot reach %s (%s)" % (base_url(cfg), exc)) from None


def probe(cfg):
    """Ask /healthz first: a mistyped address should fail before we capture."""
    request = urllib.request.Request(base_url(cfg) + "/healthz", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise PushFailed(
            "no answer from %s (%s). Check `host` and `port` in %s."
            % (base_url(cfg), getattr(exc, "reason", exc),
               uplink_config.DEFAULT_PATH)) from None


def read_jpeg():
    """One JPEG frame, or None.  The pixels are the observation (CAP-7)."""
    try:
        import cv2
    except ImportError:
        raise PushFailed("opencv is not installed, so there is no eye") from None

    cap = cv2.VideoCapture(0)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None

    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buffer.tobytes() if ok else None


def read_pcm(seconds=AUDIO_SECONDS, rate=AUDIO_RATE, channels=AUDIO_CHANNELS):
    """Raw little endian s16le samples, or None.  No volume is computed here."""
    try:
        import pyaudio
    except ImportError:
        raise PushFailed("pyaudio is not installed, so there is no ear") from None

    audio = pyaudio.PyAudio()
    chunk = 1024
    stream = audio.open(format=pyaudio.paInt16, channels=channels, rate=rate,
                        input=True, frames_per_buffer=chunk)
    frames = []
    try:
        for _ in range(int(rate / chunk * seconds)):
            # exception_on_overflow=False: a slow uplink must not crash the
            # device, it should just send a shorter sample.
            frames.append(stream.read(chunk, exception_on_overflow=False))
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
    return b"".join(frames)


def run(cfg, eye=True, ear=True, once=False, interval=None):
    """Push observations until stopped (or exactly one round with `once`)."""
    probe(cfg)
    print("device %s -> %s" % (cfg["device_id"], base_url(cfg)))

    interval = cfg["interval_s"] if interval is None else interval
    rounds = 0
    while True:
        rounds += 1
        sent = []

        if eye:
            frame = read_jpeg()
            if frame:
                push(cfg, "image/jpeg", frame)
                sent.append("eye %d bytes" % len(frame))
            else:
                print("[device] no frame from the camera")

        if ear:
            pcm = read_pcm()
            if pcm:
                push(cfg, "audio/pcm", pcm,
                     extra={"X-Sample-Rate": AUDIO_RATE,
                            "X-Channels": AUDIO_CHANNELS})
                sent.append("ear %d bytes" % len(pcm))

        print("[device] round %d: %s" % (rounds, ", ".join(sent) or "nothing to send"))

        if once:
            return rounds
        if interval > 0:
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Push a camera and a microphone to the host (CAP-6: one "
                    "direction only - nothing comes back but an ack).")
    parser.add_argument("--config", default=uplink_config.DEFAULT_PATH,
                        help="deployment file (default: %(default)s)")
    parser.add_argument("--no-eye", action="store_true",
                        help="do not use a camera")
    parser.add_argument("--no-ear", action="store_true",
                        help="do not use a microphone")
    parser.add_argument("--interval", type=float, default=None,
                        help="seconds between rounds (default: config interval_s)")
    parser.add_argument("--once", action="store_true",
                        help="one round and exit - useful to test the wiring")
    args = parser.parse_args(argv)

    try:
        cfg = uplink_config.load(args.config)
    except (OSError, ValueError) as exc:
        print("config error: %s" % exc)
        return 2

    if cfg["token"] in uplink_config.PLACEHOLDER_TOKENS:
        print("refusing to start: %r is a placeholder token, so the host "
              "would refuse every request anyway" % cfg["token"])
        return 2

    try:
        run(cfg, eye=not args.no_eye, ear=not args.no_ear,
            once=args.once, interval=args.interval)
    except PushFailed as exc:
        print("uplink failed: %s" % exc)
        return 1
    except KeyboardInterrupt:
        print()
        print("device stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
