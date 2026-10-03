"""
Seeing System - Processes visual input through camera
The AI sees the world through a camera like human eyes
"""

import json
import math
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

# Focus - R1/R2/R3, Diary/articles/2026-10-02-focus-and-fidelity.md section 5.
# Where every number below came from: parameter-values section 11 (2026-10-03).
#
# R2: how far the salience admission bar drops while attention is held on one
# place.  Reynolds & Heeger 2009 L38 reports a *reduction in contrast
# threshold* under attention, largest at intermediate contrast - that is the
# shape; it gives no number for a bar this map does not have.
# SOURCE: Carrasco (2011), *Visual attention: the past 25 years*,
#               Vision Res 51(13):1484-1525 - "For a cell to reliably detect
#               an unattended stimulus, its contrast needed to be 50% higher
#               than that of the attended stimulus; i.e., attention was
#               equivalent to about 50% increase in contrast" (Reynolds et
#               al. 2000 in V1; same order in MT and V4).  An unattended
#               cell wants 1.5x, so the attended one runs at 1/1.5: the
#               relaxation is 1 - 1/1.5 = 0.333, rounded here to 0.33.
# SWAP:       the behavioural end of the same literature is smaller -
#               Jigo & Carrasco (2020), J Vis 20(11), a 15% gain in
#               sensitivity, which would be 0.13 - and this bar is a
#               salience admission line rather than a contrast threshold,
#               so the two are not the same yardstick.  0.33 was taken on
#               the author's word, 2026-10-03; it had been 0.5 (a halving,
#               above every measurement taken) before that.
FOCUS_THRESHOLD_RELAXATION = 0.33   # 1 - 1/1.5, from Reynolds et al. 2000

# Attention counts as held once it has stayed put, and is released when it
# moves.
# SOURCE: Dugué, Merriam, Heeger & Carrasco (2020), *Differential impact of
#               endogenous and exogenous attention on activity in human
#               visual cortex*, Sci Rep 10:21274 - "endogenous takes about
#               300 ms to be deployed and can be sustained at will whereas
#               exogenous attention takes only about 100 ms to be deployed
#               and it is transient".  Müller & Rabbitt (1989), JEP:HPP
#               15:710-726 agree: peripheral cues peak within 150 ms and
#               decline between 150 and 300 ms.
# # ASSUMPTION: all three numbers below.  Ours run 1.5 s down to 0.75 s -
#               the same order and the same ordering as the measurements
#               (holding slower than releasing, 2:1 against their 3:1), but
#               longer in absolute terms: these time the build and decay of
#               a hold carried across frames, not a cue's latency.  That
#               correspondence is ours, not theirs.
#               The third one - how far the peak may travel and still count
#               as the same hold - is measured against Eriksen & Hoffman
#               (1973), Percept Psychophys 14(1):155-160, who put the
#               smallest attentional focus at about one degree of visual
#               angle.  0.02 of a typical camera's frame diagonal is close
#               to that, but the code never learns the field of view, so
#               degrees-to-pixels is an assumption on top of an assumption.
FOCUS_HOLD_TAU_S = 1.5
FOCUS_RELEASE_TAU_S = 0.75
FOCUS_HOLD_RADIUS = 0.02      # fraction of the frame diagonal


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


def _propose_regions(gray, hsv, saliency, focus):
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

    `focus` only moves the third bar (R2, Reynolds & Heeger 2009 L38:
    attention is a *reduction in contrast threshold*).  The corroboration
    rule itself never relaxes - IO-7 stands whether or not attention is
    holding, so a source still may not propose alone.

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

    # Above the map's own mean - the baseline comes from the frame itself
    # rather than from a tuned constant, the same idea the event model uses
    # when it asks whether an error is high *for it* (OPEN-21 point B) -
    # and lower still while attention is held, because R2 is that threshold
    # falling.  What it buys is exactly the noisy frame: a weak region that
    # would have died below the ordinary level is admitted at the moment
    # something is being looked at.
    bar = float(np.mean(saliency)) * (1.0 - FOCUS_THRESHOLD_RELAXATION * focus)
    mask_salient = (saliency >= bar).astype(np.uint8) * 255

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


def _attended_region(regions, saliency, center):
    """
    Which proposed region attention is on.

    The obvious answer is the one containing the map's peak.  When the peak
    fell between regions - common in a frame that is mostly noise - attention
    still lands on a thing rather than on the background, so the fallback is
    the region carrying the most salience.

    Returns an index into `regions`; `regions` is never empty here.
    """
    import numpy as np

    if center is not None:
        cx, cy = int(center[0]), int(center[1])
        for index, (box, _component) in enumerate(regions):
            x, y, w, h, _area = box
            if x <= cx < x + w and y <= cy < y + h:
                return index

    def mass(entry):
        box, component = entry
        x, y, w, h, _area = box
        roi = saliency[y:y + h, x:x + w]
        inside = component > 0
        return float(np.sum(roi[inside])) if np.any(inside) else 0.0

    return max(range(len(regions)), key=lambda i: mass(regions[i]))


def _describe_region(gray, hsv, saliency, mask, box, gabors, precision=0.0):
    """
    One region as a flat, ordered list of floats.

    Flat and unlabelled on purpose: Q1 (2026-10-02) applies the ban on
    natural language to memory itself, and this list is exactly what goes
    into `what`.  The meaning of each position lives here, in code, and is
    never written down - so there is no key to translate into a word later.

    `precision` is how much of this measurement is taken from the part of the
    region actually carrying the signal rather than from everything the
    region happens to contain (R1).  At 0 every weight below is 1 and every
    number comes out exactly as an unfocused frame produces it.

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

    # R1: how far the measurement leans on the region's own most salient
    # pixels.  0 samples the region as an even whole - today's numbers, and
    # what an unfocused frame still gets.  1 lets a pixel count for its
    # share of the region's salience, so the part of it that is standing out
    # speaks for the rest and the quiet surround stops diluting it.
    #
    # Only the scalar measurements are weighted.  The outline is never
    # touched: focus changes how the inside of a region is sampled, not
    # where the region ends, and the boundary was already decided by
    # pre-attentive contour back in proposal (IO-7).
    sal_core = roi_sal[region].astype(np.float64)
    mean_core = float(np.mean(sal_core)) if sal_core.size else 0.0
    weight = np.maximum(
        0.0,
        (1.0 - precision) + precision * sal_core / max(mean_core, 1e-6),
    )
    count = float(np.sum(weight))
    if count <= 0.0:
        weight = np.ones_like(sal_core)
        count = float(weight.size)
    if count <= 0.0:
        return None

    # Colour: what share of this region falls in each unnamed hue band, plus
    # its mean saturation and value.  Saturation and value are kept out of
    # the bands so that a desaturated region still reads as something.
    saturated = (sat >= 50) & (val >= 50)
    bands = [
        float(np.sum(weight[(hue >= lo[0]) & (hue <= hi[0]) & saturated])) / count
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
    # measured where the thing was.  This is the rest of that fix: measure
    # where the signal is, weighted by how much of it this act of attention
    # is actually holding.
    laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
    lap_region = laplacian[region]
    lap_mean = float(np.sum(weight * lap_region) / count)
    texture = float(np.sqrt(
        np.sum(weight * (lap_region - lap_mean) ** 2) / count)) / 64.0
    brightness = float(np.sum(weight * pixels) / count) / 255.0

    descriptor = list(bands)
    descriptor.append(float(np.sum(weight * sat) / count) / 255.0)
    descriptor.append(float(np.sum(weight * val) / count) / 255.0)
    descriptor.extend([circularity, aspect, area_frac, solidity])
    descriptor.append(texture)
    descriptor.append(brightness)
    for g in gabors:
        sample = g[y:y + h, x:x + w][region]
        if sample.ndim == 1:
            value = float(np.sum(weight * sample) / count)
        else:
            value = float(np.sum(weight[:, None] * sample)
                          / (count * sample.shape[1]))
        descriptor.append(value / 255.0)

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
        # Recomputed every frame from where attention has been holding, so
        # there is no standing value to load and none to set by hand.  It is
        # not written to disk either: a frame's focus belongs to that act of
        # encoding and nowhere else (R3).
        self.focus_level = 0.0  # 0=distributed, 1=held on one thing
        
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
        self._attention_center = None  # where attention sat last frame
        self._attention_at = None      # and when

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
    
    def _it_processing(self, gray, hsv, saliency, v1_features, v2_contours, v4_shapes):
        """
        IT Processing: inferotemporal cortex - where a frame becomes what the
        bot sees, with no word anywhere along the path.

        Four things happen here, each answering a different ruling.

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

        Attention (R1/R2/R3).  `focus_level` and `attention_map` were both
        present and neither decided anything; they now pick which region is
        on the spot, how far its measurement leans on its own strongest
        pixels, and how far the admission bar drops for the frame.  R3 is
        what this does *not* do: focus never reaches back into a record
        already written - induction drifts a simulation exactly the same
        amount whether or not attention was holding.

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

        regions = _propose_regions(gray, hsv, saliency,
                                   float(self.focus_level))
        if not regions:
            self.last_induction = []
            self.it_objects = []
            return []

        # One region is on the spot; the rest are measured as they always
        # were, because attention is a spotlight and not a wash.
        center = (self.attention_map or {}).get("center")
        attended = _attended_region(regions, saliency, center)

        described = []
        for index, (box, component) in enumerate(regions):
            precision = float(self.focus_level) if index == attended else 0.0
            got = _describe_region(gray, hsv, saliency, component, box,
                                   gabors, precision)
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

    def _visual_attention(self, saliency, now):
        """
        Where attention is, and how long it has been there.

        This used to be computed after IT had already described the frame -
        attention arriving after the measuring cannot guide it - and from a
        second copy of a map region proposal had computed for itself.  It now
        runs first, on the very map the regions are proposed from, and it
        decides two things: `attention_map` says *where*, `focus_level` says
        *whether it has stayed there*.

        Focus is a hold, not a reading of the picture.  The author's rule
        was that a bot focused on one thing can remember a very noisy frame,
        and a focus read off the frame's own salience would be lowest exactly
        when the frame is noisy - so focus tracks attention staying put
        instead, and rises only while the peak does not move off it.

        Guidance stays pre-attentive all the way down: the map is a
        surround difference, it carries no category and cannot name anything
        (IO-7).
        """
        import cv2

        _peak, strength, _floor, center = cv2.minMaxLoc(saliency)
        previous = self._attention_center
        last = self._attention_at

        self.attention_map = {
            "center": center,
            "strength": float(strength) / 255.0,
            "timestamp": now.isoformat(timespec="seconds"),
        }

        # Record saccade (eye movement)
        self.saccades.append({
            "from": previous if previous is not None else (0, 0),
            "to": center,
            "timestamp": now.isoformat(timespec="seconds"),
        })
        if len(self.saccades) > 50:
            self.saccades = self.saccades[-50:]

        diagonal = math.hypot(saliency.shape[1], saliency.shape[0])
        tolerance = FOCUS_HOLD_RADIUS * max(diagonal, 1.0)
        holding = False
        if previous is not None and last is not None:
            moved = math.hypot(center[0] - previous[0], center[1] - previous[1])
            holding = moved <= tolerance

        # R1 and R2 both read this: an exact exponential over real time, so
        # the value means the same thing at any frame rate.
        dt = 0.0 if last is None else max(0.0, (now - last).total_seconds())
        tau = FOCUS_HOLD_TAU_S if holding else FOCUS_RELEASE_TAU_S
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-9))
        target = 1.0 if holding else 0.0
        self.focus_level = max(0.0, min(
            1.0, self.focus_level + alpha * (target - self.focus_level)))

        self._attention_center = center
        self._attention_at = now
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
        Turn one frame into measurements: attention, then V1 -> V2 -> V4 -> IT.

        Attention comes first now and the map is computed once, because it
        used to run after IT - by which point every region had already been
        described and there was nothing left for it to guide.

        Everything here is momentary.  Q1 kept raw numbers where they belong
        - in the analysis and in the sensory ledger, never in memory - so
        `what` is the only thing that leaves this function for memory to
        see, and it is a list of numbers with no key in it.
        """
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saliency = _saliency_map(gray)

        # Attention first: it decides which region is on the spot, and IT
        # below measures on those terms.  One map, two uses, computed once.
        attention = self._visual_attention(saliency, datetime.now())

        # V1: Basic features (edges, orientation, contrast)
        v1_features = self._v1_processing(frame)

        # V2: Contours and texture
        v2_contours = self._v2_processing(frame, v1_features)

        # V4: Hue and shape, measured
        v4_shapes = self._v4_processing(frame, v1_features, v2_contours)

        # IT last: it proposes the regions, describes them, and runs
        # induction, so every measurement above is already in hand by then.
        what = self._it_processing(gray, hsv, saliency,
                                   v1_features, v2_contours, v4_shapes)

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
