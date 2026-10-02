"""
Seeing System - Processes visual input through camera
The AI sees the world through a camera like human eyes
"""

import json
import os
from datetime import datetime

# Hue bands over OpenCV's 0-179 hue range, **unnamed**.  A colour NAME is a
# word, and Q1 (2026-10-02) applied the no-words ban to all three layers at
# once - the momentary analysis, the sensory ledger, and the memory payload
# may not carry one between them.  The position in this list is an internal
# index (OPEN-21 C), never a label and never anything the memory sees.
COLOR_HUE_BANDS = [
    ([0, 50, 50], [10, 255, 255]),
    ([10, 50, 50], [25, 255, 255]),
    ([25, 50, 50], [35, 255, 255]),
    ([35, 50, 50], [85, 255, 255]),
    ([100, 50, 50], [130, 255, 255]),
    ([130, 50, 50], [170, 255, 255]),
    ([170, 50, 50], [180, 255, 255]),
]


def _color_fractions(image):
    """Share of the image covered by each hue band, in band order.

    A fixed-length list rather than a mapping: IO-6 keeps raw numbers in the
    momentary analysis and out of memory, and a list of numbers has no keys
    to name anything with.
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total = max(1, image.shape[0] * image.shape[1])
    return [
        float(np.sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0)) / total
        for lo, hi in COLOR_HUE_BANDS
    ]


# --- region proposal, descriptors, induction -------------------------------
# Diary/articles/2026-10-02-it-category-induction-design.md sections 3.1-3.5.
#
# # ASSUMPTION: these caps are ours.  Barsalou gives the induction algorithm
# and no numbers; Wolfe gives the feature classes that count as
# pre-attentive and no cutoffs for them.
MAX_REGIONS = 8
MIN_REGION_AREA = 200

# # ASSUMPTION: Barsalou offers "similar enough" as a judgement and never
# says how similar, nor how fast a simulation should drift toward the
# instance that just matched it.  Both numbers are ours.
INDUCTION_TAU = 0.85
INDUCTION_ALPHA = 0.10


def _saliency_map(gray):
    """Bottom-up salience: how far each pixel sits from its own surround."""
    import cv2

    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    return cv2.absdiff(gray, blurred)


def _similarity(a, b):
    """
    How alike two representations are: 1.0 identical, 0.0 far apart.

    Graded on purpose.  Barsalou L1286-1288 says perceptual symbol systems
    "simply assume that two similar representations are compared", and a
    comparison between two graded representations is itself graded - a set of
    tokens either matches or it does not, which is why a Jaccard distance
    could never carry this question.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    return 1.0 - sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _propose_regions(gray, hsv, saliency):
    """
    Offer the regions of a frame, with no box anywhere in the path.

    IO-7 admits only pre-attentive sources, so all three are: a closed
    contour, a saturated colour area, and salience above the frame's own
    ordinary level.  No source may propose on its own - each candidate has to
    be hit by at least two of the three - because Wolfe 2020 L107-111 puts
    contour at the bottom of the pre-attentive table, where it needs the
    other two to corroborate it.  # ASSUMPTION: "at least two of three" is
    our reading of that; the literature argues contours are weak and does
    not say how many sources must agree.

    Nothing here boxes an object: Wolfe-Horowitz L26 sets *important for
    object recognition* against *guide attention*, so a detector's box would
    be letting recognition, not attention, decide where a region starts.
    That is what NanoDet was removed for (OPEN-10 a).

    Returns [(box, mask)], box = (x, y, w, h), mask = the candidate's own
    binary mask cropped to that box.
    """
    import cv2
    import numpy as np

    # The two sources that carry shape; salience has no outline to describe.
    edges = cv2.Canny(gray, 50, 150)
    found, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounded = [c for c in found if cv2.contourArea(c) >= MIN_REGION_AREA]
    mask_bound = np.zeros(gray.shape, np.uint8)
    if bounded:
        cv2.drawContours(mask_bound, bounded, -1, 255, -1)

    mask_colour = np.zeros(gray.shape, np.uint8)
    for lo, hi in COLOR_HUE_BANDS:
        mask_colour |= cv2.inRange(hsv, np.array(lo), np.array(hi))

    # Above the map's own mean.  The baseline is taken from the frame itself
    # rather than from a tuned constant - the same idea the event model uses
    # when it asks whether an error is high *for it* (OPEN-21 point B).
    mask_salient = (saliency >= float(np.mean(saliency))).astype(np.uint8) * 255

    seed = mask_bound if np.any(mask_bound) else mask_colour
    if not np.any(seed):
        return []

    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    sources = (mask_bound, mask_colour, mask_salient)

    regions = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < MIN_REGION_AREA or w < 2 or h < 2:
            continue
        component = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        hits = sum(
            1 for source in sources
            if np.count_nonzero(component & source[y:y + h, x:x + w])
        )
        if hits < 2:
            continue
        regions.append(((x, y, w, h, area), component))

    return regions


def _describe_region(gray, hsv, saliency, mask, box, gabors):
    """
    One region as a flat, ordered list of floats.

    Flat and unlabelled on purpose: Q1 (2026-10-02) applies the ban on
    natural language to memory itself, and this list is exactly what goes
    into `what`.  The meaning of each position lives here, in code, and is
    never written down - so there is no key to translate into a word later.

    Returns (salience, descriptor), or None when the region cannot be
    described at all.
    """
    import cv2
    import numpy as np

    x, y, w, h, area = box
    region = mask > 0
    if not np.any(region):
        return None

    roi_gray = gray[y:y + h, x:x + w]
    roi_hsv = hsv[y:y + h, x:x + w]
    roi_sal = saliency[y:y + h, x:x + w]
    if roi_gray.size == 0:
        return None

    pixels = roi_gray[region]
    hsv_pixels = roi_hsv[region]
    hue = hsv_pixels[:, 0].astype(np.int32)
    sat = hsv_pixels[:, 1]
    val = hsv_pixels[:, 2]
    count = float(pixels.size)

    # Colour: what share of this region falls in each unnamed hue band, plus
    # its mean saturation and value.  Saturation and value are kept out of
    # the bands so that a desaturated region still reads as something.
    saturated = (sat >= 50) & (val >= 50)
    bands = [
        float(np.count_nonzero((hue >= lo[0]) & (hue <= hi[0]) & saturated)) / count
        for lo, hi in COLOR_HUE_BANDS
    ]

    # Shape, measured on the region's own outline rather than the frame's.
    sub = mask[y:y + h, x:x + w]
    circularity = 0.0
    solidity = 0.0
    outlines, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if outlines:
        biggest = max(outlines, key=cv2.contourArea)
        outline_area = cv2.contourArea(biggest)
        perimeter = cv2.arcLength(biggest, True)
        if perimeter > 0:
            circularity = min(1.0, 4.0 * np.pi * outline_area / (perimeter * perimeter))
        hull_area = cv2.contourArea(cv2.convexHull(biggest))
        if hull_area > 0:
            solidity = min(1.0, outline_area / hull_area)
    aspect = float(w) / float(h)
    aspect = min(aspect, 1.0 / aspect)
    area_frac = float(area) / float(w * h)

    # Texture and brightness, inside the region only.  Whole-frame averages
    # were the real gap the focus work found (R1/R2/R3): a picture that is
    # noise everywhere measured as noise everywhere, because nothing was ever
    # measured where the thing was.
    laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
    texture = float(np.std(laplacian[region])) / 64.0
    brightness = float(np.mean(pixels)) / 255.0

    descriptor = list(bands)
    descriptor.append(float(np.mean(sat)) / 255.0)
    descriptor.append(float(np.mean(val)) / 255.0)
    descriptor.extend([circularity, aspect, area_frac, solidity])
    descriptor.append(texture)
    descriptor.append(brightness)
    descriptor.extend(
        float(np.mean(g[y:y + h, x:x + w][region])) / 255.0 for g in gabors
    )

    return (float(np.mean(roi_sal[region])) / 255.0, descriptor)


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
        # Ledger: key -> {first_seen, times_seen, simulation, memory_strength}.
        # The key is an internal index and nothing else (OPEN-21 C): it holds
        # no content, it is never placed in `what`, and it means nothing
        # outside this ledger.  Categories are grown by induction, never
        # issued a name.
        self.known_objects = {}
        self._next_category = 0
        self.visual_memory = {}  # pattern -> memories
        
        # Camera state
        self.camera_available = False
        self.last_capture = None
        
        # Biological visual processing layers
        self.v1_features = []  # Basic features (V1)
        self.v2_contours = []  # Contours and texture (V2)
        self.v4_shapes = []  # Hue and shape (V4)
        self.it_objects = []  # What the frame holds, as descriptors (IT)
        # Ledger keys touched by the last induction, in the order the regions
        # came.  This is the gate's input: not what was seen but which barely
        # known corner of the ledger it landed in (CAP-7).
        self.last_induction = []
        
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
                    self._next_category = state.get("next_category", 0)
            except:
                pass
    
    def _save_state(self):
        """Save seeing state"""
        state_file = os.path.join(self.data_dir, "seeing_state.json")
        state = {
            "visual_acuity": self.visual_acuity,
            "known_objects": self.known_objects,
            "objects_recognized": self.objects_recognized,
            "next_category": self._next_category,
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
            kernel = cv2.getGaborKernel(
                (21, 21), 4.0, np.deg2rad(angle), 10.0, 0.5, 0,
                ktype=cv2.CV_32F)
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
        V4 Processing: hue and shape, measured rather than named.

        Q1 (2026-10-02) took the words out.  `circle`, `oval`,
        `large_object`, `irregular` and every colour name were categories
        from a fixed vocabulary, and a category the bot did not grow is a
        word however few of them there are.  What is left is measurement:
        where the frame sits in hue space, and how each contour measures.
        """
        color_shape = {
            "hue_fractions": _color_fractions(frame),
            "shape_features": []
        }

        for contour in v2_contours.get("contours", [])[:5]:
            color_shape["shape_features"].append({
                "area": contour["area"],
                "circularity": contour["circularity"]
            })

        self.v4_shapes = color_shape
        return color_shape
    
    def _it_processing(self, frame, v1_features, v2_contours, v4_shapes):
        """
        IT Processing: inferotemporal cortex - where a frame becomes what the
        bot sees, with no word anywhere along the path.

        Three things happen here, each answering a different ruling.

        Region proposal (IO-7).  A region is submitted when at least two of
        three pre-attentive sources hit it: a closed contour, a saturated
        colour area, and salience above the frame's own ordinary level.
        Nothing boxes anything - Wolfe-Horowitz L26 sets *important for
        object recognition* against *guide attention*, so a detector's box
        would let recognition decide where a category begins.  That is what
        NanoDet was removed for (OPEN-10 a): it carried not only COCO's
        words but COCO's boundaries.

        A descriptor per region, as a flat ordered list of floats.  It has
        no keys, so it has no name: Q1 (2026-10-02) bans natural language
        from memory itself, and this list is what goes into `what`.

        Online category induction (Barsalou 1999, L1293-1315).  Match the
        descriptor against every stored simulation; if one is similar
        enough, join that category and drift its simulation toward the
        instance that matched.  Otherwise construct a new simulation that
        matches the descriptor, and that construction establishes
        membership.  No identifier is ever issued - the judgement is that
        two representations are similar, not that two numbers are equal.

        `known_objects` is written from this and never read back into
        recognition: an entry could only exist because it was matched once,
        so reading it back to decide what is present would be the same
        question answering itself.
        """
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saliency = _saliency_map(gray)

        # Orientation, whole-frame once and sliced per region below, so that
        # every region is measured against the same four channels.
        gabors = [
            cv2.filter2D(
                gray, cv2.CV_8UC3,
                cv2.getGaborKernel((21, 21), 4.0, np.deg2rad(angle),
                                   10.0, 0.5, 0, ktype=cv2.CV_32F),
            )
            for angle in (0, 45, 90, 135)
        ]

        described = []
        for box, component in _propose_regions(gray, hsv, saliency):
            got = _describe_region(gray, hsv, saliency, component, box, gabors)
            if got is not None:
                described.append(got)

        # One ordered sequence, order = priority (OPEN-21 A).  "Recognized
        # first, appearance second" has no pre-attentive counterpart -
        # nothing here recognizes anything yet - and salience is what picks
        # first, so what would catch the eye comes first.
        described.sort(key=lambda item: -item[0])
        what = [descriptor for _salience, descriptor in described[:MAX_REGIONS]]

        self.last_induction = [self._induct(descriptor) for descriptor in what]

        self.it_objects = what
        return what

    def _induct(self, descriptor):
        """
        Barsalou's induction step, verbatim in structure (L1293-1315).

        Returns the key of the category the descriptor joined.  The key is
        an internal index and nothing else: it holds no content, is never
        put into `what`, and means nothing outside this ledger - which is
        exactly what an episodic index is (OPEN-21 C, Quest L10).
        """
        best_key = None
        best_sim = 0.0
        for key, entry in self.known_objects.items():
            simulation = entry.get("simulation") or []
            if len(simulation) != len(descriptor):
                continue
            sim = _similarity(descriptor, simulation)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key is not None and best_sim >= INDUCTION_TAU:
            entry = self.known_objects[best_key]
            simulation = entry.get("simulation") or descriptor
            entry["simulation"] = [
                (1.0 - INDUCTION_ALPHA) * old + INDUCTION_ALPHA * new
                for old, new in zip(simulation, descriptor)
            ]
            entry["times_seen"] = entry.get("times_seen", 0) + 1
            entry["memory_strength"] = min(
                1.0, entry.get("memory_strength", 0.5) + 0.05)
            return best_key

        key = str(self._next_category)
        while key in self.known_objects:
            self._next_category += 1
            key = str(self._next_category)
        self._next_category += 1
        self.known_objects[key] = {
            "first_seen": datetime.now().isoformat(),
            "times_seen": 1,
            "simulation": list(descriptor),
            "memory_strength": 0.5,
        }
        return key

    def _visual_attention(self, frame, features):
        """
        Visual Attention Mechanism
        Focus on important parts of the scene
        """
        import cv2
        import numpy as np
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # The very map region proposal used, so attention and perception look
        # at one map instead of two that merely resemble each other.
        saliency = _saliency_map(gray)
        
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

        # The ledger is already written: induction ran inside IT, while the
        # region still had its own descriptor in hand to be matched against.
        # Doing it again here would turn one observation into two memories.

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
        Turn one frame into measurements: V1 -> V2 -> V4 -> IT.

        Everything here is momentary.  Q1 kept raw numbers where they belong
        - in the analysis and in the sensory ledger, never in memory - so
        `what` is the only thing that leaves this function for memory to
        see, and it is a list of numbers with no key in it.
        """
        # V1: Basic features (edges, orientation, contrast)
        v1_features = self._v1_processing(frame)
        
        # V2: Contours and texture
        v2_contours = self._v2_processing(frame, v1_features)
        
        # V4: Hue and shape, measured
        v4_shapes = self._v4_processing(frame, v1_features, v2_contours)
        
        # IT last: it proposes the regions, describes them, and runs
        # induction, so every measurement above is already in hand by then.
        what = self._it_processing(frame, v1_features, v2_contours, v4_shapes)
        
        # Visual attention
        attention = self._visual_attention(frame, v1_features)
        
        return {
            "v1_features": v1_features,
            "v2_contours": v2_contours,
            "v4_shapes": v4_shapes,
            "what": what,
            "attention": attention,
            "brightness": v1_features.get("brightness", 0),
            "movement": False
        }
    
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
