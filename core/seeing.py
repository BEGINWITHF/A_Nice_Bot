"""
Seeing System - Processes visual input through camera
The AI sees the world through a camera like human eyes
"""

import json
import os
import time
from datetime import datetime

class SeeingSystem:
    """
    The AI's seeing system.
    Processes visual information through camera.
    Like a baby seeing the world for the first time.
    """
    
    def __init__(self, data_dir="data/senses/seeing"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Seeing state
        self.visual_acuity = 0.2  # 0=blind, 1=perfect vision
        self.focus_level = 0.5  # 0=distracted, 1=focused
        
        # What has been seen
        self.things_seen = []  # Visual observations
        self.objects_recognized = []  # Processed objects
        
        # Learning
        self.known_objects = {}  # object -> description
        
        # Camera state
        self.camera_available = False
        self.last_capture = None
        
        self._load_state()
        self._check_camera()
    
    def _load_state(self):
        """Load seeing state"""
        state_file = os.path.join(self.data_dir, "seeing_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self.visual_acuity = state.get("visual_acuity", 0.2)
                    self.known_objects = state.get("known_objects", {})
                    self.objects_recognized = state.get("objects_recognized", [])
            except:
                pass
    
    def _save_state(self):
        """Save seeing state"""
        state_file = os.path.join(self.data_dir, "seeing_state.json")
        state = {
            "visual_acuity": self.visual_acuity,
            "known_objects": self.known_objects,
            "objects_recognized": self.objects_recognized,
            "last_updated": datetime.now().isoformat()
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def _check_camera(self):
        """Check if camera is available"""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.camera_available = True
                cap.release()
            else:
                self.camera_available = False
        except ImportError:
            self.camera_available = False
            print("Warning: OpenCV not installed. Camera not available.")
            print("Install with: pip install opencv-python")
    
    def see_camera(self):
        """
        Capture a frame from the camera.
        Like opening your eyes and seeing.
        """
        if not self.camera_available:
            return {"error": "Camera not available"}
        
        try:
            import cv2
            
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Save the frame
                timestamp = int(time.time())
                filename = os.path.join(self.data_dir, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                
                # Analyze the frame
                analysis = self._analyze_frame(frame)
                
                observation = {
                    "type": "camera",
                    "filename": filename,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.things_seen.append(observation)
                self.last_capture = observation
                
                # Learn from what was seen
                if analysis:
                    for obj in analysis.get("objects", []):
                        if obj not in self.known_objects:
                            self.known_objects[obj] = {
                                "first_seen": datetime.now().isoformat(),
                                "times_seen": 1
                            }
                        else:
                            self.known_objects[obj]["times_seen"] += 1
                
                # Improve vision with use
                self.visual_acuity = min(1.0, self.visual_acuity + 0.001)
                self._save_state()
                
                return observation
            else:
                return {"error": "Failed to capture frame"}
                
        except ImportError:
            return {"error": "OpenCV not installed"}
    
    def _analyze_frame(self, frame):
        """
        Analyze a camera frame.
        Extract basic visual information.
        """
        import cv2
        import numpy as np
        
        analysis = {
            "objects": [],
            "colors": [],
            "brightness": 0,
            "movement": False
        }
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Calculate brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        analysis["brightness"] = float(np.mean(gray)) / 255.0
        
        # Detect dominant colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Simple color detection
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        if np.sum(mask) > 0:
            analysis["colors"].append("blue")
        
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        if np.sum(mask) > 0:
            analysis["colors"].append("green")
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        if np.sum(mask1) > 0 or np.sum(mask2) > 0:
            analysis["colors"].append("red")
        
        # Simple object detection using contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter significant contours
        significant_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        
        if significant_contours:
            analysis["objects"].append("shapes")
        
        return analysis
    
    def see_screenshot(self, screenshot_path):
        """
        Process a screenshot.
        Like seeing what's on a computer screen.
        """
        observation = {
            "type": "screenshot",
            "path": screenshot_path,
            "timestamp": datetime.now().isoformat()
        }
        
        self.things_seen.append(observation)
        self._save_state()
        
        return observation
    
    def set_focus(self, level):
        """Set focus level (0=distracted, 1=focused)"""
        self.focus_level = max(0.0, min(1.0, level))
        self._save_state()
    
    def get_stats(self):
        """Get seeing statistics"""
        return {
            "visual_acuity": self.visual_acuity,
            "camera_available": self.camera_available,
            "total_things_seen": len(self.things_seen),
            "objects_recognized": len(self.objects_recognized),
            "unique_objects": len(self.known_objects),
            "focus_level": self.focus_level
        }
