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
# What even counts as an observation - the filters before the gate
# ---------------------------------------------------------------------------

# A microphone reading at or below this is room tone, not a sound.
# SOURCE: the pre-existing cut in main.py's hearing branch, lifted here so a
#         local mic and a remote one cannot drift apart - they were about to
#         be two separate 0.01 literals.
# # ASSUMPTION: no measurement stands behind 0.01. It is an amplitude
#               fraction of full scale, i.e. -40 dBFS.
# SWAP:   a noise-floor reading from a real room; main.py and
#         core/host_ingest.py both read this one name.
SILENCE_FLOOR = 0.01


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

# OPEN-21 point B (2026-10-02): an early cut is not a distance from the frame
# an event started on.  It is a prediction error running above its own
# baseline, and both halves of that come straight from Kurby & Zacks (2008):
#
#   "event models ... integrating information over the recent past"        (L162)
#   "When prediction errors transiently increase relative to their
#    current baseline, event models are updated"                          (L163-164)
#
# So what is measured is (a) how far the current input sits from a model that
# keeps following the input, and (b) how far that error now sits above the
# error this very event has been running at.  The old SHOCK_THRESHOLD = 0.6
# was an absolute distance against the first frame frozen at event start,
# which is neither of those - and no person segments events that way.
#
# # ASSUMPTION: the literature gives the shape of this rule and not one of its
#               numbers.  All three below are ours.
#
# EVENT_MODEL_TAU_S: how readily the model follows the input.  Short enough
#               that a slow drift is caught up with (and therefore never
#               cuts) and long enough that one noisy frame cannot redefine
#               what is going on.
# SWAP:       raise it for a stubborn model, lower it for a skittish one.
EVENT_MODEL_TAU_S = 5.0

# SHOCK_RATIO: how far above its own baseline an error must sit to count as a
#               transient increase rather than this event's usual amount of
#               being wrong.
# SWAP:       raise it for coarser events, lower it for finer ones.
SHOCK_RATIO = 2.0

# SHOCK_FLOOR: below this an error is not a signal at all - without it a
#               baseline sitting near zero would make any wobble a shock.
# SWAP:       raise it in a noisy room.
SHOCK_FLOOR = 0.10


# ---------------------------------------------------------------------------
# Sleep rhythm - OPEN-12: the two-process model replaces the old placeholder
# ---------------------------------------------------------------------------
#
# SOURCE: Skeldon & Dijk (2025), "The two-process model of sleep regulation:
#         a mathematical perspective", npj Biol Timing Sleep 2:24 - eq. 7/8
#         and Fig. 1b; the model itself is Borbély (1982) / Daan et al. (1984).
#         local copy: papers/2025-Skeldon-Dijk-...txt
# SWAP:   TAU_*/H_*/CIRCADIAN_* are read only by the three functions at the
#         bottom of this section and by MemoryPipeline.due() - change them
#         there and nothing else in the system knows.
#
# Why these numbers and not a schedule: OPEN-13 forbids a hardcoded routine.
# "Sleep at 23:00, wake at 07:00" has no human mechanism behind it; sleep
# pressure crossing a circadian-modulated threshold does (OPEN-12).
TAU_WAKE_S = 18.2 * 3600     # chi_w: pressure rise during wake   (SOURCE T5/Fig1b)
TAU_SLEEP_S = 4.2 * 3600      # chi_s: pressure decay during sleep (SOURCE Fig1b)
                              # recovery is 4.3x faster than accumulation -
                              # that is why one night restores most of it
S_MAX = 1.0                   # mu: upper asymptote, nondimensionalised (Fig1b)
H_WAKE = 0.67                 # H0+: upper threshold, wake -> sleep (SOURCE Fig1b)
H_SLEEP = 0.17                # H0-: lower threshold, sleep -> wake (SOURCE Fig1b)
CIRCADIAN_AMPLITUDE = 0.12    # a     (SOURCE Fig1b)

# SOURCE: Czeisler et al. (1999), Science 284:2177 - 24 subjects, PCV 0.58%,
#         intrinsic period 24.18 +/- 0.04 h in both young and old.
# The model RUNS at 24.00 h because the light-dark cycle entrains it (Skeldon
# 2025 states T_c is entrained to 24 h); 24.18 h is kept as documentation of
# the free-running tendency, so a future model of drift has the value ready.
CIRCADIAN_PERIOD_S = 24.0 * 3600
CIRCADIAN_FREE_RUN_S = 24.18 * 3600
# SOURCE: Skeldon 2025 quoting young-adult alertness minimum.
# SWAP:   set this to your own chronotype - it is the ONLY thing that moves
#         the whole schedule on the clock.  With 06:00 the bot sleeps
#         01:27-09:24; with 03:00 it sleeps 22:27-06:24 (measured).
CIRCADIAN_MIN_PHASE_S = 6 * 3600


def circadian_signal(epoch_s):
    """
    C(t) in [-1, +1]: Skeldon eq. 7/8 use H(t) = H0 + a*C(t) with C(t) = cos(wt).

    +1 at the circadian maximum (MIN_PHASE + 12 h = 18:00), -1 at the
    circadian minimum (06:00).  C is the circadian drive for wakefulness, so
    both thresholds are highest when the bot is most alert and lowest in the
    biological night.
    """
    period = CIRCADIAN_PERIOD_S
    phase = (epoch_s - CIRCADIAN_MIN_PHASE_S) % period
    return math.cos(2.0 * math.pi * (phase - period / 2.0) / period)


def thresholds(epoch_s):
    """(upper H+, lower H-) at this instant.  H+ > H- always: the gap between
    them is the bistable region in which both sleep and wake can exist
    (Skeldon 2025, Fig. 3) - which is what makes 'startled awake' possible."""
    c = circadian_signal(epoch_s)
    return (H_WAKE + CIRCADIAN_AMPLITUDE * c,
            H_SLEEP + CIRCADIAN_AMPLITUDE * c)


def advance_pressure(pressure, dt, asleep):
    """
    One exact step of Process S (not Euler - the exponential is closed form).

        asleep: dP/dt = -P/tau_s      ->  P * exp(-dt/tau_s)
        awake : dP/dt = (mu - P)/tau_w -> mu + (P - mu) * exp(-dt/tau_w)

    dt <= 0 (clock moved backwards) leaves the pressure alone.
    """
    if dt <= 0:
        return pressure
    if asleep:
        value = pressure * math.exp(-dt / TAU_SLEEP_S)
    else:
        value = S_MAX - (S_MAX - pressure) * math.exp(-dt / TAU_WAKE_S)
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Sleep period behaviour - OPEN-18 (author's answers 2026-09-30)
# ---------------------------------------------------------------------------
# OPEN-18a: how often sleep runs a consolidation pass.  The author chose
# "every 90 minutes", i.e. one pass per sleep cycle rather than one per night.
# # ASSUMPTION: 90 min is the human NREM/REM cycle length; §二 of
#               parameter-values-with-sources.md lists no such constant
#               because OPEN-12 only ever specified *when* sleep starts and
#               ends, not how often it sorts during it.
# SWAP:       raise it for fewer, bigger passes; 0 disables mid-sleep passes
#             and leaves only the two boundaries (onset and waking).
SLEEP_CYCLE_S = 90 * 60

# OPEN-18b: while asleep the senses keep running (PIPE-4) but routine
# observations are NOT memorised - "常规事情感知不到".  A severe one startles
# the bot awake instead - "受到剧烈情况惊醒".
# # ASSUMPTION: no literature gives a startle threshold; 0.8 sits above the
#               0.6 that sleep spends replays on, so "worth a replay" and
#               "loud enough to wake me" are different questions.
# Reviewed by the author 2026-10-01 and deliberately left as an assumption:
#   "继续挂着" - no citation exists yet, and inventing one would be worse
#   than keeping the marker visible.
# SWAP:       any salience quantile; record() reads it once per observation.
STARTLE_SALIENCE = 0.8

# SOURCE: the author's answer 2026-09-30 ("10分钟就好").  No paper measures how
#         long a startled organism stays alert, but OPEN-13 still wants the
#         provenance - here it is a decision rather than a citation.
# Why it has to exist at all: without it the two-process model would put the
#   bot straight back to sleep.  A startled bot has pressure still above H-,
#   and while that holds, H+ says "sleep".  Skeldon 2025 calls the mechanism
#   that keeps you awake in exactly that situation "wake effort" - the upper
#   threshold is moved so that wake can be maintained.  This window is our
#   stand-in for that shift, and the gap between the thresholds (always 0.5)
#   is what makes the region between them bistable in the first place.
# SWAP: set it to 0 and the bot falls back asleep as soon as the model says
#       so; raise it for a longer alert watch (a fire keeps it up all night).
AROUSAL_S = 10 * 60

# How many sleep reports the ledger keeps.
MAX_SLEEP_REPORTS = 20
