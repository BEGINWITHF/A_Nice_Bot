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
    The AI's seeing system with biological visual processing.
    Processes visual information through camera like human visual cortex.
    Like a baby seeing the world for the first time.
    
    Biological processing pipeline:
    1. Retina (V1): Basic features - edges, contrast, brightness
    2. V2: Contours, texture, shape
    3. V4: Color processing, shape recognition
    4. IT: Object recognition and memory
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
        self.visual_memory = {}  # pattern -> memories
        
        # Camera state
        self.camera_available = False
        self.last_capture = None
        
        # Biological visual processing layers
        self.v1_features = []  # Basic features (V1)
        self.v2_contours = []  # Contours and texture (V2)
        self.v4_shapes = []  # Color and shape (V4)
        self.it_objects = []  # Object recognition (IT)
        
        # Visual attention
        self.attention_map = None  # What we're focusing on
        self.saccades = []  # Eye movements
        
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
    
    def _v1_processing(self, frame):
        """
        V1 Processing: Primary Visual Cortex
        Extracts basic features: edges, orientation, contrast
        Like the retina and V1 in the brain
        """
        import cv2
        import numpy as np
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features = {
            "brightness": float(np.mean(gray)) / 255.0,
            "contrast": float(np.std(gray)) / 128.0,
            "edges": [],
            "orientations": []
        }
        
        # Edge detection (like simple cells in V1)
        edges = cv2.Canny(gray, 50, 150)
        features["edge_density"] = float(np.sum(edges > 0)) / (edges.shape[0] * edges.shape[1])
        
        # Orientation detection (Gabor-like filters)
        for angle in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 4.0, angle, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            orientation_strength = float(np.mean(filtered)) / 255.0
            features["orientations"].append({"angle": angle, "strength": orientation_strength})
        
        self.v1_features = features
        return features
    
    def _v2_processing(self, frame, v1_features):
        """
        V2 Processing: Secondary Visual Cortex
        Processes contours, texture, and shape
        """
        import cv2
        import numpy as np
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours_data = {
            "contours": [],
            "texture": 0,
            "shape_complexity": 0
        }
        
        # Contour detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and analyze significant contours
        significant_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:  # Minimum size threshold
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    significant_contours.append({
                        "area": float(area),
                        "perimeter": float(perimeter),
                        "circularity": float(circularity),
                        "position": cv2.boundingRect(c)
                    })
        
        contours_data["contours"] = significant_contours[:10]  # Keep top 10
        contours_data["shape_complexity"] = len(significant_contours) / 10.0  # Normalize
        
        # Texture analysis using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        contours_data["texture"] = float(np.std(laplacian)) / 64.0
        
        self.v2_contours = contours_data
        return contours_data
    
    def _v4_processing(self, frame, v1_features, v2_contours):
        """
        V4 Processing: Color and Shape
        Processes color information and complex shapes
        """
        import cv2
        import numpy as np
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_shape = {
            "colors": {},
            "dominant_color": None,
            "shape_features": []
        }
        
        # Color detection (like V4 color processing)
        color_ranges = {
            "red1": ([0, 50, 50], [10, 255, 255]),
            "red2": ([170, 50, 50], [180, 255, 255]),
            "orange": ([10, 50, 50], [25, 255, 255]),
            "yellow": ([25, 50, 50], [35, 255, 255]),
            "green": ([35, 50, 50], [85, 255, 255]),
            "blue": ([100, 50, 50], [130, 255, 255]),
            "purple": ([130, 50, 50], [170, 255, 255])
        }
        
        max_color_area = 0
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            area = np.sum(mask > 0)
            if area > 0:
                color_shape["colors"][color_name] = float(area) / (frame.shape[0] * frame.shape[1])
                if area > max_color_area:
                    max_color_area = area
                    color_shape["dominant_color"] = color_name
        
        # Shape features from V2 contours
        for contour in v2_contours.get("contours", [])[:5]:
            area = contour["area"]
            circularity = contour["circularity"]
            
            # Categorize shape
            if circularity > 0.8:
                shape = "circle"
            elif circularity > 0.5:
                shape = "oval"
            elif area > 5000:
                shape = "large_object"
            else:
                shape = "irregular"
            
            color_shape["shape_features"].append({
                "shape": shape,
                "area": area,
                "circularity": circularity
            })
        
        self.v4_shapes = color_shape
        return color_shape
    
    def _it_processing(self, frame, v1_features, v2_contours, v4_shapes):
        """
        IT Processing: Inferotemporal Cortex
        Object recognition and memory integration
        """
        objects = []
        
        # Combine features for object recognition
        brightness = v1_features.get("brightness", 0)
        edge_density = v1_features.get("edge_density", 0)
        shape_complexity = v2_contours.get("shape_complexity", 0)
        dominant_color = v4_shapes.get("dominant_color")
        
        # Simple object categorization based on features
        if edge_density > 0.1 and shape_complexity > 0.3:
            objects.append("complex_object")
        elif dominant_color and v4_shapes["colors"].get(dominant_color, 0) > 0.1:
            objects.append(f"{dominant_color}_object")
        
        # Check for known objects in memory
        for obj_name, obj_data in self.known_objects.items():
            if obj_data.get("dominant_color") == dominant_color:
                objects.append(obj_name)
        
        self.it_objects = objects
        return objects
    
    def _visual_attention(self, frame, features):
        """
        Visual Attention Mechanism
        Focus on important parts of the scene
        """
        import cv2
        import numpy as np
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Saliency map (simple center-surround)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        saliency = cv2.absdiff(gray, blurred)
        
        # Find most salient region
        _, max_val, _, max_loc = cv2.minMaxLoc(saliency)
        
        # Update attention
        self.attention_map = {
            "center": max_loc,
            "strength": float(max_val) / 255.0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Record saccade (eye movement)
        self.saccades.append({
            "from": self.saccades[-1]["to"] if self.saccades else (0, 0),
            "to": max_loc,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent saccades
        if len(self.saccades) > 50:
            self.saccades = self.saccades[-50:]
        
        return self.attention_map
    
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
                
                # Learn from what was seen using IT processing
                if analysis:
                    for obj in analysis.get("it_objects", []):
                        if obj not in self.known_objects:
                            self.known_objects[obj] = {
                                "first_seen": datetime.now().isoformat(),
                                "times_seen": 1,
                                "dominant_color": analysis.get("v4_shapes", {}).get("dominant_color"),
                                "features": {
                                    "brightness": analysis.get("v1_features", {}).get("brightness", 0),
                                    "edge_density": analysis.get("v1_features", {}).get("edge_density", 0),
                                    "shape_complexity": analysis.get("v2_contours", {}).get("shape_complexity", 0)
                                }
                            }
                        else:
                            self.known_objects[obj]["times_seen"] += 1
                            # Strengthen memory with each viewing (Hebbian learning)
                            if "memory_strength" not in self.known_objects[obj]:
                                self.known_objects[obj]["memory_strength"] = 0.5
                            self.known_objects[obj]["memory_strength"] = min(1.0, 
                                self.known_objects[obj]["memory_strength"] + 0.05)
                
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
        Analyze a camera frame using biological visual processing pipeline.
        Processes through V1 -> V2 -> V4 -> IT like the human brain.
        """
        # V1: Basic features (edges, orientation, contrast)
        v1_features = self._v1_processing(frame)
        
        # V2: Contours and texture
        v2_contours = self._v2_processing(frame, v1_features)
        
        # V4: Color and shape
        v4_shapes = self._v4_processing(frame, v1_features, v2_contours)
        
        # IT: Object recognition
        it_objects = self._it_processing(frame, v1_features, v2_contours, v4_shapes)
        
        # Visual attention
        attention = self._visual_attention(frame, v1_features)
        
        analysis = {
            "v1_features": v1_features,
            "v2_contours": v2_contours,
            "v4_shapes": v4_shapes,
            "it_objects": it_objects,
            "attention": attention,
            "objects": it_objects,
            "colors": list(v4_shapes.get("colors", {}).keys()),
            "brightness": v1_features.get("brightness", 0),
            "movement": False
        }
        
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
            "focus_level": self.focus_level,
            "visual_memory_size": len(self.visual_memory)
        }
    
    def consolidate_visual_memory(self):
        """
        Visual Memory Consolidation
        Like sleep-dependent memory consolidation in the brain
        Reviews and strengthens important visual memories
        """
        consolidated = 0
        
        for obj_name, obj_data in self.known_objects.items():
            times_seen = obj_data.get("times_seen", 0)
            
            # Calculate memory importance
            importance = times_seen / 10.0  # More seen = more important
            
            # Apply spaced repetition
            if "memory_strength" not in obj_data:
                obj_data["memory_strength"] = 0.5
            
            # Strengthen based on importance and recency
            if importance > 0.5:
                obj_data["memory_strength"] = min(1.0, 
                    obj_data["memory_strength"] + 0.01 * importance)
                consolidated += 1
            
            # Decay weak memories
            if obj_data["memory_strength"] < 0.1 and times_seen < 3:
                # Mark for forgetting (but don't delete immediately)
                obj_data["memory_strength"] *= 0.9
        
        self._save_state()
        return consolidated
    
    def recognize_object(self, frame):
        """
        Recognize objects in a frame using learned patterns
        Like IT cortex matching against stored memories
        """
        import cv2
        import numpy as np
        
        # Process through visual pipeline
        v1 = self._v1_processing(frame)
        v2 = self._v2_processing(frame, v1)
        v4 = self._v4_processing(frame, v1, v2)
        
        recognized = []
        
        # Compare against known objects
        for obj_name, obj_data in self.known_objects.items():
            similarity = 0
            
            # Color similarity
            if v4.get("dominant_color") == obj_data.get("dominant_color"):
                similarity += 0.4
            
            # Brightness similarity
            brightness_diff = abs(v1.get("brightness", 0) - obj_data.get("features", {}).get("brightness", 0))
            similarity += max(0, 0.3 - brightness_diff)
            
            # Memory strength affects recognition
            memory_strength = obj_data.get("memory_strength", 0.5)
            similarity *= memory_strength
            
            if similarity > 0.3:
                recognized.append({
                    "object": obj_name,
                    "confidence": similarity,
                    "times_seen": obj_data.get("times_seen", 0)
                })
        
        return recognized
