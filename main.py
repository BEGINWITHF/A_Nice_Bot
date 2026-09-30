"""
Pure AI - Sensory Experience Only
The AI experiences the world ONLY through hearing and seeing
No text input, no text output - like a real human baby
"""

import cv2
import numpy as np
import os
import time
import sys
import threading
from datetime import datetime
from core.sensory import SensorySystem
from core.pure_learning import PureLearningSystem
from core.memory_pipeline import MemoryPipeline, visual_salience, audio_salience
from core.parameters import SILENCE_FLOOR
from core.host_ingest import HostIngest
from core.uplink import ObservationServer, API_VERSION
import uplink_config

# Initialize systems
senses = SensorySystem()
brain = PureLearningSystem()
pipeline = MemoryPipeline()

# STEP-2 / CAP-6: the uplink server runs on its own thread and shares this
# brain with the sensing loop.  Everything that touches the three memory
# layers, the sensory ledgers or the state files they write goes through
# here, on both threads, so one is never mid-sleep while the other records.
INGEST_LOCK = threading.RLock()


def register_stores(pipeline, senses, brain):
    """
    Attach every long term store to the memory pipeline.

    Each store must expose forget() that really deletes - a store which only
    lowers a strength value never forgets anything.
    """
    pipeline.register(
        "hearing",
        consolidate=senses.hearing.consolidate_auditory_memory,
        forget=senses.hearing.forget_pass,
    )
    pipeline.register(
        "seeing",
        consolidate=senses.seeing.consolidate_visual_memory,
        forget=senses.seeing.forget_pass,
    )
    pipeline.register("learning", forget=brain.forget_pass)
    pipeline.register("sensory", forget=senses.forget_pass)

class MicrophoneInput:
    """Handles microphone input for hearing"""
    def __init__(self):
        self.available = False
        try:
            import pyaudio
            self.pyaudio = pyaudio
            self.audio = pyaudio.PyAudio()
            self.chunk = 1024
            self.format = pyaudio.paInt16
            self.channels = 1
            self.rate = 44100
            self.available = True
        except ImportError:
            print("pyaudio not installed. Install with: pip install pyaudio")
    
    def listen(self, duration=2):
        """Listen for sounds"""
        if not self.available:
            return None
        
        try:
            stream = self.audio.open(format=self.format,
                                   channels=self.channels,
                                   rate=self.rate,
                                   input=True,
                                   frames_per_buffer=self.chunk)
            
            frames = []
            for _ in range(0, int(self.rate / self.chunk * duration)):
                data = stream.read(self.chunk)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            
            # Analyze sound
            volume = np.abs(audio_data).mean() / 32768.0
            return {"volume": float(volume), "data": audio_data}
        except Exception as e:
            return None

class CameraInput:
    """Handles camera input for seeing"""
    def __init__(self):
        self.available = False
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.available = True
                cap.release()
        except:
            pass
    
    def capture(self):
        """Capture a frame"""
        if not self.available:
            return None
        
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame
        except:
            pass
        return None


def sense_once(camera, mic):
    """
    One pass of both senses, then a sleep check if one is due.

    Everything the brain touches runs under INGEST_LOCK, because the uplink
    server feeds the same pipeline from its own thread: the three memory
    layers, the sensory ledgers and every state file they write are shared,
    and one thread must not be mid-sleep while the other records a frame.
    """
    with INGEST_LOCK:
        # 1. SEE - Camera captures environment
        if camera.available:
            # SensorySystem opens the camera and analyses the frame itself,
            # so there is no second capture here - one frame per cycle.
            result = senses.see_camera()

            # Salience gate (CAP-7): the host decides what is worth
            # keeping, a frame with nothing new is not memorised
            if result and "analysis" in result:
                objects = result["analysis"].get("it_objects") or []
                pipeline.record(
                    "seeing",
                    {"objects": sorted(objects)},
                    salience=visual_salience(objects, senses.seeing.known_objects),
                )

        # 2. HEAR - Microphone captures sounds
        if mic.available:
            sound = mic.listen(duration=1)
            if sound and sound["volume"] > SILENCE_FLOOR:  # above room tone
                # Process through sensory system
                result = senses.hear_sound("sound", volume=sound["volume"])

                # AI's internal processing (silent)
                brain.hear_word("sound")
                # Salience gate: room tone is not a memory, a loud sound is
                volume = sound["volume"]
                pipeline.record(
                    "hearing",
                    {"volume_band": int(volume * 20)},
                    salience=audio_salience(volume),
                )

        # 3. SLEEP WHEN DUE - one pass, never blocks the sensing loop.
        # OPEN-12: `due()` is Process S crossing a circadian-modulated
        # threshold, or a 90-minute sleep cycle while already asleep.
        if pipeline.due():
            report = pipeline.sleep(reason="scheduled")
            print()
            print(f"  {report['state']}: {report['transition']} "
                  f"pressure={report['pressure']} "
                  f"replayed={report['replayed']} "
                  f"dropped={report['dropped_below_threshold']} "
                  f"recent={report['recent_out']} "
                  f"episodes={report['episodes']}")


def start_uplink():
    """
    Bring up the CAP-5 HTTP uplink, or explain why not and carry on.

    Two ways to end up disabled: the port cannot be bound, or the token is
    still the placeholder from the example file.  Both are operator fixes,
    and neither should take the bot down with it - it is meant to be resident
    24/7 (GOAL-2), so local sensing keeps running either way.
    """
    cfg = None
    try:
        cfg = uplink_config.load()
    except (OSError, ValueError) as exc:
        # A hand-edited file with a trailing comma must not stop the bot.
        print("Uplink:  DISABLED - cannot read configs/uplink.json (%s)" % exc)
        return None

    if cfg["token"] in uplink_config.PLACEHOLDER_TOKENS:
        print("Uplink:  DISABLED - %r is a placeholder token" % cfg["token"])
        print("         put a random string in configs/uplink.json, or delete")
        print("         the file so a real one gets generated")
        return None

    server = ObservationServer(
        HostIngest(senses, brain, pipeline, INGEST_LOCK),
        bind=cfg["bind"],
        port=cfg["port"],
        token=cfg["token"],
    )
    try:
        host, port = server.start()
    except (OSError, ValueError) as exc:
        print("Uplink:  DISABLED - %s" % exc)
        print("         fix configs/uplink.json and restart")
        return None

    print("Uplink:  %s:%s  POST /%s/observations" % (host, port, API_VERSION))
    print("         set `host` in each device's copy of configs/uplink.json")
    return server


def main():
    """
    Main loop - AI experiences world through senses only
    No text input, no text output
    """
    # Initialize sensory inputs
    mic = MicrophoneInput()
    camera = CameraInput()
    register_stores(pipeline, senses, brain)
    
    print("PURE AI - Sensory Experience")
    print("=" * 40)
    print("Microphone:", "Ready" if mic.available else "Not available")
    print("Camera:", "Ready" if camera.available else "Not available")
    print()
    print("The AI is experiencing the world...")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    print()

    uplink = start_uplink()
    print()

    # Main sensory loop
    try:
        cycle = 0
        while True:
            cycle += 1
            
            sense_once(camera, mic)

            # 4. REST - Brief pause between sensing
            time.sleep(0.5)
            
            # Print minimal status (no internal thoughts)
            if cycle % 20 == 0:  # Every 10 seconds
                with INGEST_LOCK:
                    stats = senses.get_overall_state()
                    # advances Process S as a side effect
                    state = pipeline.state()
                sys.stdout.write(f"\r{state}: p={pipeline.pressure:.2f} | ")
                sys.stdout.write(f"Awareness: {stats['awareness_level']:.2f} | ")
                sys.stdout.write(f"Load: {stats['sensory_load']:.2f} | ")
                sys.stdout.write(f"Cycle: {cycle}")
                sys.stdout.flush()
            
    except KeyboardInterrupt:
        print()
        print()
        print("Stopping sensory experience...")
        print("The AI keeps its memories.")
    finally:
        # Close the uplink first: a frame still in flight must not record
        # into a pipeline that is being torn down.  Then save under the same
        # lock the sensing loop used, and do it on any exit, not just Ctrl+C -
        # a stray exception used to skip the save entirely.
        if uplink is not None:
            uplink.stop()
        with INGEST_LOCK:
            pipeline.sleep(reason="shutdown")
            senses._save_state()
            brain._save_state()

if __name__ == "__main__":
    main()
