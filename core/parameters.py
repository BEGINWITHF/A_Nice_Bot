"""
Every number the memory system uses, in one file.  (Clause OPEN-13)

Why this module exists
-----------------------
OPEN-13: each value carries its provenance next to it, so a whole source can
be swapped in one place instead of hunting through the code.  The three-part
record required by OPEN-13 is:

    value   -> the constant below
    source  -> "# SOURCE:" line, plus Diary/articles/2026-09-29-
               parameter-values-with-sources.md section number
    swap    -> "# SWAP:" line: what to change if the source is replaced

Anything with no literature behind it is marked "# ASSUMPTION:" instead of
being dressed up as a citation (DATA-6 / OPEN-13).

Clause DATA-6: where psychology measured something, the value comes from that
measurement.  Nothing here is stored *inside* a memory entry (DATA-4/5) -
these are the rules about memories, not memories.
"""

import math

# ---------------------------------------------------------------------------
# Forgetting curve - OPEN-14 / OPEN-15  (parameter-values section 3)
# ---------------------------------------------------------------------------
# Two-component ("MCM") fit to Ebbinghaus' savings data.
#
# SOURCE: Murre & Dros (2015), "Replication and Analysis of Ebbinghaus'
#         Forgetting Curve", PLoS ONE 10(5):e0120644, Table 5 ("MCM, Ebbinghaus")
#         local copy: papers/2015-Murre-Dros-...txt (values read from the
#         table image papers/Murre2015-Table5-MCM-fits.png)
#         time unit = seconds
# SWAP:   replace A1/MU2/A2 with any other published fit of the same shape;
#         accessibility() is the only function that reads them, so the rest
#         of the system does not change.
#
# Q(t) = e^(-a1 t) + [mu2/(a1-a2)] * (e^(-a2 t) - e^(-a1 t))
# normalised so that accessibility(0) == 1 (the paper's Q(0) = mu1 = 0.704,
# which cancels out when we divide by it - see parameter-values section 3).
#
# # ASSUMPTION: the *shape* is sourced (Murre & Dros), the *level* is
#               calibrated so that accessibility(0) = 1.  Savings != recall,
#               so 1.0 here means "fully available", not "100% recall
#               accuracy" - parameter-values section 6 warns about this.
A1 = 0.000319      # fast component, per second   (SOURCE: Murre 2015 T5)
MU2 = 0.000145     # slow component weight        (SOURCE: Murre 2015 T5)
A2 = 1.79e-7       # slow component, per second   (SOURCE: Murre 2015 T5)

# Points of the curve worth knowing when reading a report
# (computed from the constants above, kept as comments for eyeballing):
#   10 min -> 0.905    1 h -> 0.627    6 h -> 0.454    1 day -> 0.448
#   forget_threshold 0.08 with no load -> ~112 days


def accessibility(elapsed_seconds, load=0.0):
    """
    How available a memory still is, 0..1.

    `elapsed_seconds` is time since its last rehearsal; `load` is how crowded
    the layer it lives in is (entries / capacity).  Soft capacity (OPEN-15 D,
    from Bays et al. 2009's shared-resource model): a crowded layer makes
    every memory *effectively older*, instead of kicking one entry out.

        t_eff = elapsed * (1 + load)

    # ASSUMPTION: Bays measured precision allocation at a single moment;
    #             applying it to the decay rate is an extrapolation.
    #             Marked here so it can be found and removed later.
    """
    t = max(0.0, float(elapsed_seconds)) * (1.0 + max(0.0, float(load)))
    if t <= 0.0:
        return 1.0
    e1 = math.exp(-A1 * t)
    e2 = math.exp(-A2 * t)
    value = e1 + (MU2 / (A1 - A2)) * (e2 - e1)
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Layers - OPEN-14 / OPEN-15 (B: focus / recent / long term)
# ---------------------------------------------------------------------------

# SOURCE: Cowan (2001), "The magical number 4 in short-term memory...",
#         Behavioral and Brain Sciences 24:87-185 - capacity of the activated
#         focus is 3-5 items; OPEN-14 rounds up to 6 ("slightly more than
#         human", author's answer 2026-09-29).
# SWAP:   any better estimate of focus capacity; the layer is only read here.
CONSCIOUS_CAPACITY = 6

# # ASSUMPTION: no literature gives a single number for the layer below the
#              focus.  The author approved starting with 65 on 2026-09-29
#              ("first use 65, mark # ASSUMPTION:").  Direction is sourced:
#              far above Cowan's 4-6, far below unlimited.  See
#              parameter-values section 4.
# SWAP:     change this one number here; nothing else hard-codes it.
RECENT_CAPACITY = 65

# Budget for the long-term episode store, chosen to match the other long
# term stores (they all cap at 500).
# # ASSUMPTION: engineering budget, not a psychological quantity.
LONG_TERM_BUDGET = 500


# ---------------------------------------------------------------------------
# Forgetting thresholds - OPEN-14 / OPEN-15
# ---------------------------------------------------------------------------

# SOURCE: OPEN-13 answer - "use the most authoritative data you can find and
#         record the source so it can be swapped".  Chosen so that a memory
#         that is never rehearsed survives ~112 days (no load) on the MCM
#         curve above; see parameter-values section 3.
# SWAP:   any published recall-accuracy floor; only sleep() reads it.
FORGET_THRESHOLD = 0.08

# Salience needed for sleep to replay an entry.
# SOURCE: OPEN-15 evidence - Rasch & Born (2013) require consolidation to be
#         selective (otherwise "system overflow"); the numeric cut is ours.
# # ASSUMPTION: the 0.6 cut-off itself has no measurement behind it.
# SWAP:   any salience quantile you like; replay() and transfer both read it.
REPLAY_SALIENCE = 0.6

# An entry must be at least this old before sleep transfers it to long term.
# SOURCE: Cepeda et al. (2006) meta-analysis of distributed practice -
#         spacing of at least ~1 day is where the benefit of a second
#         exposure becomes robust; parameter-values section 3.
# SWAP:   any other published optimal spacing; sleep() reads it once.
PROMOTE_MIN_AGE_S = 86400.0    # 1 day


# ---------------------------------------------------------------------------
# Event segmentation - OPEN-16 (decided: G = window + content shock)
# ---------------------------------------------------------------------------

# Default length of one event: if nothing has been observed for this long,
# the next observation opens a new event.
#
# # ASSUMPTION: four measured sources converge near 30 s, none of them says
#               "30" - Zacks, Tversky & Iyer (2001) coarse 34.3 s / fine
#               12.8 s; Zacks et al. (2010) coarse 59.4 s / fine 22.9 s;
#               Geerligs et al. (2022) neural states 4.5-27.2 s, keypress
#               boundaries 6.5-93.7 s; Shim et al. (2022) paradigm 22.5-31.5 s;
#               plus Cowan (2001) activation maintenance 2-30 s.
#               The semantic is "default cut on timeout", NOT "an event
#               ends every 30 s" - Gomez et al. (2025) showed duration alone
#               predicts everyday event endings at only 4-5% accuracy.
#               Full three-part record: parameter-values section 8.
# SWAP:       any single measured mean; only segment() reads it.
WINDOW_S = 30.0

# How different the content must be for a cut to happen *before* the window
# expires (the G in OPEN-16 G: "window first, content shock cuts early").
# Measured as 1 - Jaccard similarity over the *set* fields of the payload.
#
# # ASSUMPTION: no literature gives a shock threshold for this comparison.
#               The direction is sourced - Zacks & Swallow (2007) "when a
#               salient feature changes" and Zacks et al. (2010) situation
#               changes predict segmentation - but 0.6 is ours.
# SWAP:       raise it for coarser events, lower it for finer ones.
SHOCK_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Rhythm (still the placeholder from before OPEN-12; see design-map)
# ---------------------------------------------------------------------------
# # ASSUMPTION: kept as-is until OPEN-12 (two-process sleep) is implemented;
#              OPEN-12 already decided to replace these with tau_w/tau_s and
#              the circadian thresholds.
SLEEP_AFTER_SECONDS = 900
SLEEP_AFTER_EVENTS = 200

# How many sleep reports the ledger keeps.
MAX_SLEEP_REPORTS = 20
