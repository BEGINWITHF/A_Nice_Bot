"""
Where a remote observation becomes a perception - the host half of CAP-7.

CAP-7 splits the work down the middle: the device is a dumb sense organ that
ships the raw picture or the raw waveform, and decoding, perception and the
salience gate all belong to the host (依据 PIPE-2 - when to memorise is the
host memory pipeline's decision, so a device may not judge for itself).

The author settled the ambiguity on 2026-09-30 when asked how raw "原始
观测" has to be: "在 ai 眼里他应该是能看到远端摄像机的画面（画面被实时传回
服务器），而服务器再让 ai 感知" - the picture comes back to the server in real
time and the server is what lets the AI perceive it.  So the pixels cross the
wire and the perception happens here, through exactly the same
`seeing.see_frame()` the local camera uses.

Nothing is stored as a picture either: see_frame() turns the frame into
features and drops it (IO-5), so a remote camera cannot smuggle a JPEG into
the bot's memory by arriving from somewhere else.

Like the rest of `core/`, opencv and numpy are imported where they are used
rather than at module level - that is what lets the test suite run on a bare
interpreter with nothing but pytest installed.
"""

import threading

from core.memory_pipeline import audio_salience, visual_salience
from core.parameters import SILENCE_FLOOR
from core.uplink import UplinkError


def _vision_libs():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        # Told to the device rather than swallowed: an uplink that answers
        # 200 for a frame nobody looked at would be a silent lie.
        raise UplinkError(503, "this host cannot see: %s" % exc) from None
    return cv2, np


def _audio_lib():
    try:
        import numpy as np
    except ImportError as exc:
        raise UplinkError(503, "this host cannot hear: %s" % exc) from None
    return np


class HostIngest:
    """
    Callable for `core.uplink.ObservationServer(deliver=...)`.

    `lock` serialises against the local sensing loop: the pipeline, the
    sensory ledgers and their state files are shared, and one thread must not
    be writing a JSON file while the other is mid-sleep.
    """

    def __init__(self, senses, brain, pipeline, lock=None):
        self.senses = senses
        self.brain = brain
        self.pipeline = pipeline
        self.lock = lock if lock is not None else threading.RLock()

    def __call__(self, channel, body, meta):
        with self.lock:
            if channel == "vision":
                return self._vision(body, meta)
            if channel == "sound":
                return self._sound(body, meta)
        raise UplinkError(415, "unsupported channel: %r" % (channel,))

    def _vision(self, body, meta):
        cv2, np = _vision_libs()
        frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            # A device that ships a truncated frame gets told so, rather than
            # having its failure recorded as a memory of nothing.
            raise UplinkError(400, "not a decodable jpeg frame")

        device = meta.get("device") or "unnamed"
        result = self.senses.see_frame(frame, source="camera:%s" % device)
        if not result or "analysis" not in result:
            raise UplinkError(422, "frame could not be analysed")

        objects = result["analysis"].get("it_objects") or []

        # CAP-7: the gate is here, on the host, in the same place a frame
        # from the local camera meets it.  The device never sees the result.
        entry = self.pipeline.record(
            "seeing",
            {"objects": sorted(objects)},
            salience=visual_salience(objects, self.senses.seeing.known_objects),
        )
        return {"channel": "vision", "recorded": entry is not None,
                "objects": sorted(objects)}

    def _sound(self, body, meta):
        np = _audio_lib()
        if len(body) % 2:
            raise UplinkError(400, "pcm body must be an even number of bytes")
        samples = np.frombuffer(body, dtype="<i2")
        if samples.size == 0:
            raise UplinkError(400, "empty audio body")

        volume = float(np.abs(samples).mean()) / 32768.0

        # Room tone is not an observation.  The device transmits regardless -
        # CAP-7 forbids it deciding what is worth sending - so the filter
        # lives here, where the local loop applies it too.
        if volume <= SILENCE_FLOOR:
            return {"channel": "sound", "recorded": False, "silence": True}

        self.senses.hear_sound("sound", volume=volume)
        self.brain.hear_word("sound")

        entry = self.pipeline.record(
            "hearing",
            {"volume_band": int(volume * 20)},
            salience=audio_salience(volume),
        )
        return {"channel": "sound", "recorded": entry is not None}
