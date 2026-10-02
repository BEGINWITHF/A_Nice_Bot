"""
Tests for the capability part (STEP-2).

Design clauses:
    CAP-5  HTTP/REST over TCP out of the standard library
    CAP-6  v1 is one direction: device -> host, the reply is an ack
    CAP-7  the device ships raw bytes; perception and the gate stay on host
    CAP-8  the host address is configured by hand
    OPEN-7 a shared bearer token, industry standard (author, 2026-09-30)
    IO-5   a frame is sampled and dropped - never written down

Run:
    python -m pytest test_uplink.py -v
"""

import json
import os
import socket
import struct
import sys
import urllib.error
import urllib.request

import pytest

# The transport half (CAP-5/CAP-6/CAP-8, OPEN-7) is standard library only and
# runs anywhere.  The perception half (CAP-7) needs the two libraries the bot
# itself needs, and skips rather than fails when they are missing.
try:
    import numpy as np
except ImportError:                     # pragma: no cover - bare interpreter
    np = None
try:
    import cv2
except ImportError:                     # pragma: no cover - bare interpreter
    cv2 = None

needs_numpy = pytest.mark.skipif(np is None, reason="numpy is not installed")
needs_cv2 = pytest.mark.skipif(cv2 is None, reason="opencv is not installed")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uplink_config
from core.host_ingest import HostIngest
from core.memory_pipeline import MemoryPipeline
from core.seeing import SeeingSystem
from core.sensory import SensorySystem
from core.uplink import (
    HEALTH_PATH,
    OBSERVATION_PATH,
    ObservationServer,
    UplinkError,
)

TOKEN = "a-token-nobody-could-guess"
DEVICE = "desk-eye"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(url):
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def post(url, body, headers=None):
    request = urllib.request.Request(
        url, data=body, headers=headers or {}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def raw(headers, body=b"", port=0, host="127.0.0.1"):
    """
    Send a request written by hand, so a missing or absurd Content-Length
    can be tested - urllib always sets one correctly.
    """
    text = headers if headers.endswith("\r\n\r\n") else headers + "\r\n\r\n"
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(text.encode("ascii") + body)
        chunks = []
        try:
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
        except OSError:
            pass
    return b"".join(chunks)


def status_of(response):
    return int(response.split(b" ", 2)[1])


def jpeg(green_box=True):
    """A real JPEG - imdecode on the host must have something to decode."""
    assert cv2 is not None and np is not None, "opencv and numpy are required"
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    if green_box:
        frame[20:70, 20:70] = (0, 255, 0)
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return buffer.tobytes()


def pcm(volume, samples=4410):
    """Raw little endian s16le at a fixed amplitude - standard library only."""
    return struct.pack("<%dh" % samples,
                       *([int(32767 * volume)] * samples))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def deliveries():
    return []


@pytest.fixture
def server(deliveries):
    def deliver(channel, body, meta):
        deliveries.append((channel, body, meta))
        return {"recorded": True}

    instance = ObservationServer(
        deliver, bind="127.0.0.1", port=0, token=TOKEN, max_body=65536)
    instance.start()
    yield instance
    instance.stop()


@pytest.fixture
def base(server):
    host, port = server.address
    return "http://127.0.0.1:%s" % port


@pytest.fixture
def host(tmp_path):
    """A whole host with a permissive gate, for the CAP-7 half."""
    pipeline = MemoryPipeline(data_dir=str(tmp_path / "mem"),
                              policy={"min_salience": 0.0})
    senses = SensorySystem(str(tmp_path / "senses"))
    ingest = HostIngest(senses, pipeline)
    ingest.spy = []
    real = pipeline.record

    def spy(kind, payload, salience=0.5, now=None):
        ingest.spy.append((kind, payload, salience))
        return real(kind, payload, salience=salience, now=now)

    pipeline.record = spy
    return ingest


def observation_headers(content_type, token=TOKEN, **extra):
    headers = {
        "Content-Type": content_type,
        "Authorization": "Bearer %s" % token,
        "X-Device-Id": DEVICE,
    }
    headers.update(extra)
    return headers


# ---------------------------------------------------------------------------
# CAP-5 / CAP-6 / OPEN-7: the transport
# ---------------------------------------------------------------------------

def test_healthz_answers_without_a_token(base):
    status, _, body = get(base + HEALTH_PATH)
    assert status == 200
    assert json.loads(body) == {"ok": True}


def test_a_post_without_a_token_is_refused(base, deliveries):
    status, headers, body = post(
        base + OBSERVATION_PATH, b"\xff\xd8nobody",
        {"Content-Type": "image/jpeg"})

    assert status == 401
    # RFC 6750: tell the client which scheme it failed, nothing more.
    assert headers["WWW-Authenticate"].startswith("Bearer")
    assert json.loads(body)["ok"] is False
    assert deliveries == []


def test_a_wrong_token_is_refused(base, deliveries):
    status, _, _ = post(
        base + OBSERVATION_PATH, b"\xff\xd8nobody",
        observation_headers("image/jpeg", token="not-the-token"))

    assert status == 401
    assert deliveries == []


def test_a_frame_is_handed_to_deliver_verbatim(base, deliveries):
    # The transport never looks inside the body - that is CAP-7's job.
    frame = b"\xff\xd8\xff\xe0raw-jpeg-bytes"
    status, _, body = post(
        base + OBSERVATION_PATH, frame,
        observation_headers("image/jpeg", **{"X-Captured-At": "1759250000.5"}))

    assert status == 200
    assert json.loads(body) == {"ok": True}
    assert deliveries == [
        ("vision", frame,
         {"device": DEVICE, "captured_at": 1759250000.5,
          "sample_rate": None, "channels": None}),
    ]


def test_the_reply_says_nothing_about_the_gate(base, deliveries):
    """CAP-7: the device must not learn what the host judged worth keeping."""
    _, _, body = post(
        base + OBSERVATION_PATH, b"\xff\xd8nobody",
        observation_headers("image/jpeg"))

    assert set(json.loads(body)) == {"ok"}


def test_audio_metadata_reaches_deliver(base, deliveries):
    body = pcm(0.5)
    status, _, _ = post(
        base + OBSERVATION_PATH, body,
        observation_headers(
            "audio/pcm", **{"X-Sample-Rate": "44100", "X-Channels": "1"}))

    assert status == 200
    channel, payload, meta = deliveries[0]
    assert channel == "sound"
    assert payload == body
    assert meta["sample_rate"] == 44100
    assert meta["channels"] == 1


def test_an_unsupported_media_type_is_refused(base, deliveries):
    status, _, body = post(
        base + OBSERVATION_PATH, b"whatever",
        observation_headers("application/json"))

    assert status == 415
    assert "image/jpeg" in json.loads(body)["supported"]
    assert deliveries == []


def test_a_missing_content_length_is_refused(base, server):
    """Without it the body has no defined end on a keep-alive connection."""
    _, port = server.address
    response = raw(
        "POST %s HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Authorization: Bearer %s\r\n"
        "Content-Type: image/jpeg" % (OBSERVATION_PATH, TOKEN), port=port)

    assert status_of(response) == 411


def test_an_oversized_body_is_refused_before_it_is_read(base, server):
    _, port = server.address
    response = raw(
        "POST %s HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Authorization: Bearer %s\r\n"
        "Content-Type: image/jpeg\r\n"
        "Content-Length: 99999999" % (OBSERVATION_PATH, TOKEN), port=port)

    assert status_of(response) == 413


def test_an_unknown_path_is_404(base):
    status, _, _ = post(
        base + "/v1/commands", b"", observation_headers("image/jpeg"))
    assert status == 404


def test_an_unsupported_method_is_405_and_says_what_is_allowed(base):
    request = urllib.request.Request(
        base + OBSERVATION_PATH, data=b"", method="PUT")
    try:
        urllib.request.urlopen(request, timeout=5)
        pytest.fail("PUT must not be accepted")
    except urllib.error.HTTPError as exc:
        assert exc.code == 405
        assert exc.headers["Allow"] == "GET, POST"


def test_the_server_refuses_to_start_without_a_token(deliveries):
    with pytest.raises(ValueError):
        ObservationServer(lambda *a: None, bind="127.0.0.1", port=0, token="")


def test_a_deliver_rejection_is_reported_with_its_status(base, deliveries):
    def reject(channel, body, meta):
        raise UplinkError(422, "frame could not be analysed")

    instance = ObservationServer(reject, bind="127.0.0.1", port=0, token=TOKEN)
    instance.start()
    try:
        host, port = instance.address
        status, _, body = post(
            "http://127.0.0.1:%s%s" % (port, OBSERVATION_PATH),
            b"\xff\xd8nobody", observation_headers("image/jpeg"))
    finally:
        instance.stop()

    assert status == 422
    assert json.loads(body)["error"] == "frame could not be analysed"


def test_a_deliver_crash_becomes_500_and_the_server_survives(base):
    instance = ObservationServer(
        lambda *a: (_ for _ in ()).throw(RuntimeError("boom")),
        bind="127.0.0.1", port=0, token=TOKEN)
    instance.start()
    try:
        host, port = instance.address
        url = "http://127.0.0.1:%s%s" % (port, OBSERVATION_PATH)
        status, _, _ = post(url, b"x", observation_headers("image/jpeg"))
        # One bad frame must not take the host down with it.
        assert status == 500
        status, _, _ = post(
            url, b"x", observation_headers("audio/pcm"))
        assert status == 500
        assert get("http://127.0.0.1:%s%s" % (port, HEALTH_PATH))[0] == 200
    finally:
        instance.stop()


def test_rejected_requests_are_counted(base, server):
    post(base + OBSERVATION_PATH, b"x", {"Content-Type": "image/jpeg"})
    post(base + OBSERVATION_PATH, b"x", observation_headers("image/jpeg"))

    assert server.stats == {"received": 1, "rejected": 1}


# ---------------------------------------------------------------------------
# CAP-7: the host is what perceives
# ---------------------------------------------------------------------------

@needs_cv2
def test_a_remote_frame_is_seen_through_the_local_eye(host):
    seen_before = len(host.senses.seeing.things_seen)

    result = host("vision", jpeg(), {"device": DEVICE})

    seen = host.senses.seeing.things_seen
    assert len(seen) == seen_before + 1
    # One consciousness (DATA-4): a remote eye is recorded as an eye, with
    # only its provenance telling it apart from the local camera.
    assert seen[-1]["type"] == "camera:%s" % DEVICE
    assert result["channel"] == "vision"
    assert isinstance(result["regions"], int)


@needs_cv2
def test_a_remote_frame_reaches_the_pipeline_with_a_salience(host):
    host("vision", jpeg(), {"device": DEVICE})

    assert [kind for kind, _, _ in host.spy] == ["seeing"]
    _, payload, salience = host.spy[0]
    assert set(payload) == {"what"}
    assert 0.0 <= salience <= 1.0


@needs_cv2
def test_a_corrupt_frame_is_rejected_and_never_memorised(host):
    seen_before = len(host.senses.seeing.things_seen)

    with pytest.raises(UplinkError) as excinfo:
        host("vision", b"certainly not a jpeg", {"device": DEVICE})

    assert excinfo.value.status == 400
    assert len(host.senses.seeing.things_seen) == seen_before
    assert host.spy == []


@needs_numpy
def test_room_tone_from_a_remote_mic_is_not_memorised(host):
    """The device transmits regardless (CAP-7); the filter lives here."""
    result = host("sound", pcm(0.0), {"device": DEVICE})

    assert result["silence"] is True
    assert host.spy == []


@needs_numpy
def test_a_loud_remote_sample_is_memorised(host):
    result = host("sound", pcm(0.5), {"device": DEVICE})

    assert result["recorded"] is True
    assert [kind for kind, _, _ in host.spy] == ["hearing"]
    # A "half volume" sample reads 16383/32768, so it lands in band 9.
    assert host.spy[0][1] == {"volume_band": 9}
    assert host.spy[0][2] == pytest.approx(1.0)


@needs_cv2
def test_a_remote_frame_leaves_no_pixels_behind(host):
    """IO-5: whoever captured the frame, it is sampled and dropped."""
    host("vision", jpeg(), {"device": DEVICE})

    leftovers = [name for root, _, files in os.walk(os.path.dirname(str(host.pipeline.data_dir)))
                 for name in files
                 if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png")]
    assert leftovers == []


def test_the_local_loop_and_an_uplink_share_one_lock(tmp_path):
    """Nothing may record while another thread is mid-sleep."""
    import threading

    lock = threading.RLock()
    ingest = HostIngest(
        SensorySystem(str(tmp_path / "s")),
        MemoryPipeline(data_dir=str(tmp_path / "m"), policy={"min_salience": 0.0}),
        lock,
    )
    assert ingest.lock is lock


# ---------------------------------------------------------------------------
# see_frame(): capture and perception, split for CAP-7
# ---------------------------------------------------------------------------

@needs_cv2
def test_see_frame_records_which_eye_it_came_through(tmp_path):
    seeing = SeeingSystem(str(tmp_path))

    observation = seeing.see_frame(
        np.zeros((64, 64, 3), dtype=np.uint8), source="camera:%s" % DEVICE)

    assert observation["type"] == "camera:%s" % DEVICE
    assert seeing.things_seen[-1] is observation
    assert seeing.last_capture is observation
    assert "analysis" in observation


def test_see_camera_still_reports_a_missing_eye(tmp_path):
    seeing = SeeingSystem(str(tmp_path))     # _check_camera is patched out
    assert seeing.see_camera() == {"error": "Camera not available"}


@needs_numpy
def test_a_remote_frame_lifts_the_body_like_the_local_eye(tmp_path):
    senses = SensorySystem(str(tmp_path))
    load_before, awareness_before = senses.sensory_load, senses.awareness_level

    senses.see_frame(np.zeros((64, 64, 3), dtype=np.uint8),
                     source="camera:%s" % DEVICE)

    assert senses.sensory_load == pytest.approx(load_before + 0.2)
    assert senses.awareness_level == pytest.approx(awareness_before + 0.1)


# ---------------------------------------------------------------------------
# CAP-8 / OPEN-7: the deployment file
# ---------------------------------------------------------------------------

def test_a_first_run_creates_a_config_with_a_real_token(tmp_path):
    path = str(tmp_path / "uplink.json")

    cfg = uplink_config.load(path, quiet=True)

    assert os.path.exists(path)
    assert cfg["token"] not in uplink_config.PLACEHOLDER_TOKENS
    assert len(cfg["token"]) >= 32
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle)["token"] == cfg["token"]


def test_a_missing_key_falls_back_to_its_default(tmp_path):
    path = tmp_path / "uplink.json"
    path.write_text(json.dumps({"token": "hand-written"}), encoding="utf-8")

    cfg = uplink_config.load(str(path), quiet=True)

    assert cfg["token"] == "hand-written"
    assert cfg["port"] == uplink_config.DEFAULTS["port"]
    assert cfg["bind"] == uplink_config.DEFAULTS["bind"]


def test_a_bad_port_is_reported_not_accepted(tmp_path):
    path = tmp_path / "uplink.json"
    path.write_text(json.dumps({"token": "x", "port": "eight"}),
                    encoding="utf-8")

    with pytest.raises(ValueError):
        uplink_config.load(str(path), quiet=True)


def test_the_committed_example_is_not_a_usable_secret():
    with open(uplink_config.EXAMPLE_PATH, encoding="utf-8") as handle:
        example = json.load(handle)

    assert example["token"] in uplink_config.PLACEHOLDER_TOKENS


def test_the_example_documents_every_key_the_code_knows():
    with open(uplink_config.EXAMPLE_PATH, encoding="utf-8") as handle:
        example = json.load(handle)

    assert set(example) == set(uplink_config.DEFAULTS)


# ---------------------------------------------------------------------------
# CAP-8: what a device says when the hand-typed address is wrong
# ---------------------------------------------------------------------------

def write_config(tmp_path, **overrides):
    cfg = dict(uplink_config.DEFAULTS)
    cfg["token"] = TOKEN
    cfg["host"] = "127.0.0.1"
    cfg.update(overrides)
    path = tmp_path / "uplink.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_a_device_reaches_a_live_host_without_any_hardware(tmp_path, server):
    import device

    _, port = server.address
    path = write_config(tmp_path, port=port)

    assert device.main(["--config", path, "--no-eye", "--no-ear",
                        "--once"]) == 0


def test_a_device_refuses_the_placeholder_token(tmp_path):
    import device

    path = tmp_path / "uplink.json"
    path.write_text(json.dumps({"token": "CHANGE_ME"}), encoding="utf-8")

    assert device.main(["--config", str(path), "--no-eye", "--no-ear",
                        "--once"]) == 2


def test_a_device_explains_a_mistyped_address(tmp_path, capsys):
    import device

    path = write_config(tmp_path, port=free_port())

    assert device.main(["--config", path, "--no-eye", "--no-ear",
                        "--once"]) == 1
    # CAP-8 addresses are typed by hand, so the failure says which key to fix.
    out = capsys.readouterr().out
    assert "`host`" in out and "`port`" in out


def test_a_device_explains_a_token_mismatch(tmp_path):
    import device

    wrong = ObservationServer(
        lambda *a: None, bind="127.0.0.1", port=0, token="the-hosts-token")
    wrong.start()
    try:
        cfg = dict(uplink_config.DEFAULTS, token="not-the-hosts-token",
                   port=wrong.address[1])
        with pytest.raises(device.PushFailed) as excinfo:
            device.push(cfg, "image/jpeg", b"\xff\xd8frame")
    finally:
        wrong.stop()

    # The fix is a shared file, so the message names the file.
    assert "configs/uplink.json" in str(excinfo.value)
    assert "token" in str(excinfo.value)


def test_a_device_stamps_every_observation_with_when_it_was_taken():
    import device

    cfg = dict(uplink_config.DEFAULTS, token=TOKEN, device_id="desk-eye")
    headers = device.headers_for(cfg, "image/jpeg")

    assert headers["Content-Type"] == "image/jpeg"
    assert headers["Authorization"] == "Bearer %s" % TOKEN
    assert headers["X-Device-Id"] == "desk-eye"
    # The pipeline windows observations on a clock (OPEN-16), so the host
    # must learn when the sample happened, not when it arrived.
    assert float(headers["X-Captured-At"]) > 0
