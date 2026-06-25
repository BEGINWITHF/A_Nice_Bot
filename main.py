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
from datetime import datetime
from core.sensory import SensorySystem
from core.pure_learning import PureLearningSystem

# Initialize systems
senses = SensorySystem()
brain = PureLearningSystem()

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

def main():
    """
    Main loop - AI experiences world through senses only
    No text input, no text output
    """
    # Initialize sensory inputs
    mic = MicrophoneInput()
    camera = CameraInput()
    
    print("PURE AI - Sensory Experience")
    print("=" * 40)
    print("Microphone:", "Ready" if mic.available else "Not available")
    print("Camera:", "Ready" if camera.available else "Not available")
    print()
    print("The AI is experiencing the world...")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    print()
    
    # Main sensory loop
    try:
        cycle = 0
        while True:
            cycle += 1
            
            # 1. SEE - Camera captures environment
            if camera.available:
                frame = camera.capture()
                if frame is not None:
                    # Process through sensory system
                    result = senses.see_camera()
                    
                    # AI's internal processing (silent)
                    if result and "analysis" in result:
                        # Internal state updates
                        brain.seeing._analyze_frame(frame)
            
            # 2. HEAR - Microphone captures sounds
            if mic.available:
                sound = mic.listen(duration=1)
                if sound and sound["volume"] > 0.01:  # Above silence threshold
                    # Process through sensory system
                    result = senses.hear_sound("sound", volume=sound["volume"])
                    
                    # AI's internal processing (silent)
                    brain.hear_word("sound")
            
            # 3. REST - Brief pause between sensing
            time.sleep(0.5)
            
            # Print minimal status (no internal thoughts)
            if cycle % 20 == 0:  # Every 10 seconds
                stats = senses.get_overall_state()
                sys.stdout.write(f"\rAwareness: {stats['awareness_level']:.2f} | ")
                sys.stdout.write(f"Load: {stats['sensory_load']:.2f} | ")
                sys.stdout.write(f"Cycle: {cycle}")
                sys.stdout.flush()
            
    except KeyboardInterrupt:
        print()
        print()
        print("Stopping sensory experience...")
        print("The AI keeps its memories.")
        
        # Save state
        senses._save_state()
        brain._save_state()

if __name__ == "__main__":
    main()
