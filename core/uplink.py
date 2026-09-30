"""
Uplink: the host half of the capability part (STEP-2).

CAP-5  - HTTP/REST over TCP, served out of the Python standard library, so a
         self-hosting user installs nothing extra (that was one of CAP-5's two
         criteria) and the wire protocol survives a change of language - a
         Rust, C, MCU or browser client all ship an HTTP stack already.
CAP-6  - v1 is one direction only, device -> host.  The response is a bare
         acknowledgement; commands come later, stacked on top as a WebSocket
         without this being thrown away.
CAP-7  - the device is a dumb sense organ.  It ships the raw picture or the
         raw waveform; decoding, perception and the salience gate all happen
         behind `deliver`, on the host.  Settled by the author 2026-09-30:
         "在 ai 眼里他应该是能看到远端摄像机的画面（画面被实时传回服务器），
         而服务器再让 ai 感知" - the pixels cross the wire, nothing else does.
CAP-8  - a device finds the host by a manually configured address; see
         uplink_config.py.

Auth is the industry standard the author asked for when OPEN-7 was raised
(2026-09-30: "你按照行业标准就好"): an RFC 6750 bearer token in
`Authorization`, compared in constant time.  HTTPS is deliberately not part
of v1 - a self-signed certificate on a home LAN makes CAP-8's manual
addressing worse, not better - and is the remaining piece of OPEN-7 if the
port is ever reachable beyond the LAN.

This module imports the standard library and nothing else, so the transport
can be exercised without a camera, a microphone or a brain.
"""

import hmac
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_VERSION = "v1"
OBSERVATION_PATH = "/%s/observations" % API_VERSION
HEALTH_PATH = "/healthz"

# The channel is the Content-Type, so a client needs no schema beyond the
# media types below.  `audio/pcm` is little endian signed 16 bit, interleaved.
VISION_TYPES = ("image/jpeg", "image/jpg")
SOUND_TYPES = ("audio/pcm",)

# A frame is ~100 KB and two seconds of 44.1 kHz s16le mono is ~176 KB, so
# 8 MB of headroom still cannot let one peer fill the host's memory.
DEFAULT_MAX_BODY = 8 * 1024 * 1024


class UplinkError(Exception):
    """
    Raised by `deliver` to reject an observation with a specific HTTP status.

    Anything else escaping `deliver` becomes a 500: one malformed frame must
    never take the host down with it.
    """

    def __init__(self, status, message):
        super().__init__(message)
        self.status = int(status)
        self.message = str(message)


def _channel_for(content_type):
    """Map a Content-Type onto a channel name, or None if we do not take it."""
    value = (content_type or "").split(";", 1)[0].strip().lower()
    if value in VISION_TYPES:
        return "vision"
    if value in SOUND_TYPES:
        return "sound"
    return None


def _header_int(headers, name):
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _header_float(headers, name):
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


class _Handler(BaseHTTPRequestHandler):
    """
    One request, one observation.

    The gate outcome is never written into the response: CAP-7 says the device
    does not get to learn what is worth remembering, so the reply stays a bare
    `{"ok": true}` no matter what the pipeline decided.
    """

    server_version = "NiceBotUplink/%s" % API_VERSION
    protocol_version = "HTTP/1.1"

    # The default access log would put a line on stdout for every frame, which
    # is noise in a 24/7 bot (GOAL-2).  Failures still get one short line,
    # because CAP-8 addresses are typed by hand and get mistyped.
    def log_message(self, fmt, *args):
        try:
            code = args[1] if len(args) > 1 else None
            if code is not None and int(code) >= 400:
                sys.stderr.write(
                    "[uplink] %s %s\n" % (self.address_string(), fmt % args))
        except (IndexError, TypeError, ValueError):
            pass

    def _send_json(self, status, payload, headers=None, close=False):
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.command == "POST":
            stats = getattr(self.server, "uplink_stats", None)
            if stats is not None:
                # Requests arrive on their own threads, so the counter needs
                # the same courtesy the rest of the host gets.
                with self.server.uplink_stats_lock:
                    stats["rejected" if status >= 400 else "received"] += 1
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if close:
            self.close_connection = True
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(blob)

    def _bearer_ok(self):
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        presented = header[len("Bearer "):].strip().encode("utf-8")
        expected = self.server.uplink_token.encode("utf-8")
        # A token is a credential, not a hint - compare in constant time so
        # the response does not leak how many leading characters matched.
        return hmac.compare_digest(presented, expected)

    # -- GET ---------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == HEALTH_PATH:
            # CAP-6 keeps v1 to one direction, but a liveness probe is not a
            # command: CAP-8 addresses are configured by hand, and being able
            # to ask "is this the right port?" is what makes that bearable.
            return self._send_json(200, {"ok": True})
        return self._send_json(404, {"ok": False, "error": "not found"})

    # -- POST --------------------------------------------------------------

    def do_POST(self):
        if self.path.split("?", 1)[0] != OBSERVATION_PATH:
            return self._send_json(404, {"ok": False, "error": "not found"},
                                   close=True)

        # Authenticate before reading a byte of the body: an unauthenticated
        # peer should cost the host as little as possible.
        if not self._bearer_ok():
            return self._send_json(
                401,
                {"ok": False, "error": "unauthorised"},
                headers={"WWW-Authenticate": 'Bearer realm="%s"' % API_VERSION},
                close=True,
            )

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            # Every client CAP-5 names - Rust, C, MCU, browser - sets this,
            # and without it the body has no defined end on a keep-alive
            # connection, so we refuse rather than guess.
            return self._send_json(411, {"ok": False, "error": "length required"},
                                   close=True)
        try:
            length = int(raw_length)
        except ValueError:
            return self._send_json(400, {"ok": False, "error": "bad content length"},
                                   close=True)
        if length < 0:
            return self._send_json(400, {"ok": False, "error": "bad content length"},
                                   close=True)

        # Check the declared size before reading anything.
        if length > self.server.uplink_max_body:
            return self._send_json(413, {"ok": False, "error": "body too large"},
                                   close=True)

        channel = _channel_for(self.headers.get("Content-Type"))
        if channel is None:
            # Drain the body so this connection stays usable, then refuse.
            self.rfile.read(length)
            return self._send_json(
                415,
                {"ok": False, "error": "unsupported media type",
                 "supported": list(VISION_TYPES + SOUND_TYPES)},
            )

        body = self.rfile.read(length)
        if len(body) != length:
            return self._send_json(400, {"ok": False, "error": "truncated body"},
                                   close=True)

        meta = {
            "device": (self.headers.get("X-Device-Id") or "").strip() or "unnamed",
            "captured_at": _header_float(self.headers, "X-Captured-At"),
            "sample_rate": _header_int(self.headers, "X-Sample-Rate"),
            "channels": _header_int(self.headers, "X-Channels"),
        }

        try:
            self.server.uplink_deliver(channel, body, meta)
        except UplinkError as exc:
            return self._send_json(exc.status, {"ok": False, "error": exc.message},
                                   close=exc.status >= 500)
        except Exception as exc:            # noqa: BLE001
            sys.stderr.write("[uplink] deliver failed: %r\n" % (exc,))
            return self._send_json(500, {"ok": False, "error": "internal error"},
                                   close=True)

        return self._send_json(200, {"ok": True})

    def _method_not_allowed(self):
        # Anything else would otherwise answer 501, which says "not
        # implemented yet" when the truth is "not allowed here".
        self._send_json(405, {"ok": False, "error": "method not allowed"},
                        headers={"Allow": "GET, POST"}, close=True)

    do_PUT = do_DELETE = do_PATCH = _method_not_allowed


class ObservationServer:
    """
    A threaded HTTP server that hands every observation to `deliver`.

    `deliver(channel, body, meta)` runs on the server's own thread; the caller
    is responsible for synchronising it against anything else that touches the
    brain.  HostIngest in core/host_ingest.py takes the lock for you.
    """

    def __init__(self, deliver, bind="0.0.0.0", port=8765, token="",
                 max_body=DEFAULT_MAX_BODY):
        if not callable(deliver):
            raise TypeError("deliver must be callable")
        if not token:
            # An empty token would compare equal to an empty header, i.e. the
            # door would be open.  Refuse at construction, not at request time.
            raise ValueError("an empty bearer token leaves the uplink open")
        self._deliver = deliver
        self.bind = str(bind)
        self.port = int(port)
        self.token = str(token)
        self.max_body = int(max_body)
        self._httpd = None
        self._thread = None

    @property
    def running(self):
        return self._httpd is not None

    @property
    def address(self):
        """The bound (host, port) - useful when the configured port was 0."""
        if self._httpd is None:
            return None
        host, port = self._httpd.server_address[:2]
        return host, port

    @property
    def stats(self):
        if self._httpd is None:
            return {"received": 0, "rejected": 0}
        return dict(self._httpd.uplink_stats)

    def start(self):
        if self._httpd is not None:
            raise RuntimeError("uplink server is already running")
        httpd = ThreadingHTTPServer((self.bind, self.port), _Handler)
        httpd.daemon_threads = True       # never hold the process open (GOAL-2)
        httpd.uplink_deliver = self._deliver
        httpd.uplink_token = self.token
        httpd.uplink_max_body = self.max_body
        httpd.uplink_stats = {"received": 0, "rejected": 0}
        httpd.uplink_stats_lock = threading.Lock()
        self._httpd = httpd
        self._thread = threading.Thread(
            # A tight poll interval keeps stop() snappy: shutdown() only
            # notices the flag on the next tick.  One select() per 50 ms is
            # nothing for a process that is meant to sit resident all day.
            target=lambda: httpd.serve_forever(poll_interval=0.05),
            name="uplink", daemon=True)
        self._thread.start()
        return self.address

    def stop(self):
        httpd, thread = self._httpd, self._thread
        self._httpd = None
        self._thread = None
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
