"""
One JSON file describes an uplink deployment.

CAP-8  - a device finds the host by a manually configured address; `host` is
         that address, and this file is the manual configuration.
OPEN-7 - the shared bearer token lives here too.  The author settled the
         question on 2026-09-30 with "你按照行业标准就好", so it is an
         RFC 6750 token that both sides read from the same file.

Both halves read the same shape:

    host side uses     bind, port, token
    device side uses   host, port, token, device_id, interval_s

The real file is gitignored because of the token; the committed
`configs/uplink.example.json` documents the keys, and `load()` copies it into
place on first run, replacing the placeholder with a freshly generated
secret.  Copy the resulting file to each device and set `host` to the address
that machine can actually reach.
"""

import json
import os
import secrets
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT, "configs")
EXAMPLE_PATH = os.path.join(CONFIG_DIR, "uplink.example.json")
DEFAULT_PATH = os.path.join(CONFIG_DIR, "uplink.json")

# Only a fallback: a file copied to a device on its own, or written by an
# older revision, is merged over these rather than rejected.
DEFAULTS = {
    "bind": "0.0.0.0",
    "port": 8765,
    "host": "127.0.0.1",
    "token": "",
    "device_id": "device-1",
    "interval_s": 1.0,
}

# A token nobody changed is worse than no token, because it looks configured.
PLACEHOLDER_TOKENS = ("", "CHANGE_ME", "changeme", "replace-me")


def load(path=DEFAULT_PATH, create=True, quiet=False):
    """
    Read the deployment file, creating it from the example on first run.

    Raises FileNotFoundError when the file is absent and `create` is False,
    and ValueError when it exists but is not a JSON object.
    """
    if not os.path.exists(path):
        if not create:
            raise FileNotFoundError(path)
        _create(path, quiet=quiet)

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("%s must contain a JSON object" % path)

    cfg = dict(DEFAULTS)
    cfg.update({key: value for key, value in data.items() if key in DEFAULTS})

    try:
        cfg["port"] = int(cfg["port"])
        cfg["interval_s"] = float(cfg["interval_s"])
    except (TypeError, ValueError) as exc:
        raise ValueError("%s: port and interval_s must be numbers" % path) from exc
    if cfg["port"] < 0 or cfg["port"] > 65535:
        raise ValueError("%s: port %r is not a valid port" % (path, cfg["port"]))

    if str(cfg["token"]) in PLACEHOLDER_TOKENS and not quiet:
        print("[uplink] WARNING: %s still holds a placeholder token. The "
              "host refuses to start on one." % path)
    return cfg


def _create(path, quiet=False):
    """Copy the example into place and give it a real token."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    token = secrets.token_hex(24)      # 48 hex chars = 192 bits

    if os.path.exists(EXAMPLE_PATH):
        shutil.copyfile(EXAMPLE_PATH, path)
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except ValueError:
            cfg = dict(DEFAULTS)
        if not isinstance(cfg, dict):
            cfg = dict(DEFAULTS)
    else:
        cfg = dict(DEFAULTS)

    cfg["token"] = token
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    if not quiet:
        print("[uplink] created %s with a generated token" % path)
        print("[uplink] copy it to each device and set `host` to this "
              "machine's address.")
    return path


if __name__ == "__main__":
    # `python uplink_config.py` prints the deployment the host will use.
    config = load()
    shown = dict(config)
    shown["token"] = "%s... (%d chars)" % (shown["token"][:6], len(shown["token"]))
    print(json.dumps(shown, indent=2, ensure_ascii=False))
