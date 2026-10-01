"""
Seeing System - Processes visual input through camera
The AI sees the world through a camera like human eyes
"""

import json
import os
from datetime import datetime

# Named colour ranges - the vocabulary V4 speaks.  Module level so that the
# whole-frame pass and one detected object's crop are described with the same
# words: a colour name in known_objects only means anything if both sides used
# these ranges.
COLOR_RANGES = {
    "red1": ([0, 50, 50], [10, 255, 255]),
    "red2": ([170, 50, 50], [180, 255, 255]),
    "orange": ([10, 50, 50], [25, 255, 255]),
    "yellow": ([25, 50, 50], [35, 255, 255]),
    "green": ([35, 50, 50], [85, 255, 255]),
    "blue": ([100, 50, 50], [130, 255, 255]),
    "purple": ([130, 50, 50], [170, 255, 255]),
}


def _color_fractions(image):
    """Share of the image covered by each named colour range."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total = max(1, image.shape[0] * image.shape[1])
    fractions = {}
    for name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        area = np.sum(mask > 0)
        if area > 0:
            fractions[name] = float(area) / total
    return fractions


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

        # Object recognition (OPEN-10).  Constructing it is free: nothing is
        # downloaded and no cv2 is imported until the first frame arrives.
        from .vision_model import Detector
        self.detector = Detector()
        self.it_detections = []  # richest form of the last IT output

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
        color_shape = {
            "colors": {},
            "dominant_color": None,
            "shape_features": []
        }

        # Color detection (like V4 color processing)
        color_shape["colors"] = _color_fractions(frame)
        if color_shape["colors"]:
            color_shape["dominant_color"] = max(
                color_shape["colors"], key=color_shape["colors"].get)

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

        OPEN-10, the author's answer of 2026-10-01: patch perception quality
        first and wire the learning up afterwards, one sense at a time, vision
        first.  "Patched" means IT names what is actually in the frame rather
        than inferring a name from edge density and the frame's dominant
        colour.

        The old version also read names back out of known_objects and appended
        any whose stored colour matched the frame's dominant colour.  That was
        circular: an entry could only be in known_objects because IT had named
        it once, and because the colour was measured over the whole frame,
        every entry carrying it came back on every frame whether or not the
        object was still there.  It is gone - known_objects is written from
        detections and is never read back into recognition.
        """
        detections = self.detector.detect(frame)
        self.it_detections = detections

        objects = []
        if self.detector.available:
            # One entry per name, strongest first: the label is what gets
            # stored, so repeating it here would only inflate times_seen.
            # An empty list is a real answer - the detector ran and the desk
            # really does hold no person - so the heuristics below must not
            # get a second vote and invent something it just denied.
            for detection in sorted(detections, key=lambda d: -d["confidence"]):
                if detection["label"] not in objects:
                    objects.append(detection["label"])
            self.it_objects = objects
            return objects

        # No detector - no cv2, no weights, or no network.  Fall back to the
        # feature heuristics rather than go blind, and leave the reason in
        # self.detector.error so a caller can log it instead of guessing.
        edge_density = v1_features.get("edge_density", 0)
        shape_complexity = v2_contours.get("shape_complexity", 0)
        dominant_color = v4_shapes.get("dominant_color")

        if edge_density > 0.1 and shape_complexity > 0.3:
            objects.append("complex_object")
        elif dominant_color and v4_shapes["colors"].get(dominant_color, 0) > 0.1:
            objects.append(f"{dominant_color}_object")

        self.it_objects = objects
        return objects
    
    def _object_color(self, label, frame, detections, fallback):
        """
        The colour of one object, not of whatever it happened to be standing in.

        The ledger used to store the whole frame's dominant colour under each
        object, so two unrelated things caught in the same view came back
        sharing a colour - and that shared colour is exactly what the old
        recognition loop read back to declare them the same object.  A
        detection gives us a box, so we can simply look at the box.
        """
        for detection in detections:
            if detection.get("label") != label:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in detection["box"])
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            fractions = _color_fractions(crop) if crop.size else {}
            if fractions:
                return max(fractions, key=fractions.get)
            # Clearly seen, but it carries no named colour of its own.
            return None
        # Heuristic labels have no box: they came out of the frame, so the
        # frame's colour is the only one they can honestly be given.
        return fallback

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
    
    def see_frame(self, frame, source="camera"):
        """
        Turn one frame into an observation.  This is where vision happens.

        CAP-7 splits capture from perception: a remote device ships the raw
        picture to the host and the host is what perceives it - the author's
        answer when asked how raw "raw observation" has to be (2026-09-30):
        "画面被实时传回服务器，而服务器再让 ai 感知".  So the local camera and
        an uplink frame land here through one code path, and `source` only
        records which eye it came through.

        IO-5 still holds for both: the frame is sampled into features and
        dropped, so the pixels are never written down by whoever captured them.
        """
        # IO-5: sample the frame, turn it into features, drop the
        # pixels. The raw image is never written to disk - a JPEG is
        # not a memory, the analysis below is.
        analysis = self._analyze_frame(frame)

        observation = {
            "type": source,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

        self.things_seen.append(observation)
        self.last_capture = observation

        # Learn from what was seen using IT processing
        if analysis:
            detections = analysis.get("detections", [])
            frame_color = analysis.get("v4_shapes", {}).get("dominant_color")
            for obj in analysis.get("it_objects", []):
                if obj not in self.known_objects:
                    self.known_objects[obj] = {
                        "first_seen": datetime.now().isoformat(),
                        "times_seen": 1,
                        "dominant_color": self._object_color(
                            obj, frame, detections, frame_color),
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

    def see_camera(self):
        """
        Capture a frame from the camera, then see it.
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
                return self.see_frame(frame, source="camera")
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
            "detections": list(self.it_detections),
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

    def forget_pass(self, max_objects=500, max_things=600, max_recognized=600,
                    forget_threshold=0.05):
        """
        Sleep-time pruning, called by MemoryPipeline.sleep().

        consolidate_visual_memory() only lowers memory_strength, and strength
        multiplied by 0.9 never reaches zero - so it never deletes anything.
        This method does the actual deleting.
        """
        dropped = {"known_objects": 0, "things_seen": 0, "objects_recognized": 0,
                   "visual_memory": 0}

        # Objects that were seen once, long ago, and never impressed
        before = len(self.known_objects)
        weak = [
            name for name, d in self.known_objects.items()
            if d.get("times_seen", 0) < 3
            and d.get("memory_strength", 0.5) < forget_threshold
        ]
        for name in weak:
            del self.known_objects[name]
        dropped["known_objects"] = before - len(self.known_objects)

        if len(self.known_objects) > max_objects:
            ranked = sorted(
                self.known_objects.items(),
                key=lambda kv: (kv[1].get("memory_strength", 0.5),
                                kv[1].get("times_seen", 0)),
                reverse=True,
            )
            keep = dict(ranked[:max_objects])
            dropped["known_objects"] += len(self.known_objects) - len(keep)
            self.known_objects = keep

        # Raw observation buffers: vision is not a video recorder
        if len(self.things_seen) > max_things:
            dropped["things_seen"] = len(self.things_seen) - max_things
            self.things_seen = self.things_seen[-max_things:]
        if len(self.objects_recognized) > max_recognized:
            dropped["objects_recognized"] = len(self.objects_recognized) - max_recognized
            self.objects_recognized = self.objects_recognized[-max_recognized:]

        if len(self.visual_memory) > max_objects:
            ranked = sorted(
                self.visual_memory.items(),
                key=lambda kv: kv[1].get("strength", 0.5)
                if isinstance(kv[1], dict) else 0.5,
                reverse=True,
            )
            keep = dict(ranked[:max_objects])
            dropped["visual_memory"] = len(self.visual_memory) - len(keep)
            self.visual_memory = keep

        if any(dropped.values()):
            self._save_state()
        return dropped
