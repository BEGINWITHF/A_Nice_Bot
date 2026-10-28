"""
Barsalou's online category induction, shared by both senses.

Diary/articles/2026-10-02-it-category-induction-design.md sections 3.1-3.5.
Barsalou (1999) *Perceptual Symbol Systems*, L1293-1315: match a descriptor
against every stored simulation; if one is similar enough, join that
category and drift its simulation toward the instance that matched.
Otherwise construct a new simulation that matches the descriptor, and that
construction establishes membership.  No identifier is ever issued - the
judgement is that two representations are similar, not that two numbers are
equal.

Vision has always run this rule, and hearing must run the same one rather
than a second copy of it: OPEN-22 D8 gives hearing exactly the same `what`
shape vision has (a flat ordered list of floats with no name in it), so the
categorisation step is not sense-specific.  One rule, one file, two callers.

A key is a content-free index.  That is what OPEN-21 C ruled an episodic
index to be (Quest L10: *holds no specific information and is agnostic to
its contents*) - it may name a row and it may never name a thing.
"""

# # ASSUMPTION: Barsalou offers "similar enough" as a judgement and never
# says how similar, nor how fast a simulation should drift toward the
# instance that just matched it.  Both numbers are ours.
INDUCTION_TAU = 0.85
INDUCTION_ALPHA = 0.10


def similarity(descriptor, simulation):
    """
    How alike two representations are: 1.0 identical, 0.0 far apart.

    Graded on purpose.  Barsalou L1286-1288 says perceptual symbol systems
    "simply assume that two similar representations are compared", and a
    comparison between two graded representations is itself graded - a set of
    tokens either matches or it does not, which is why a Jaccard distance
    could never carry this question.

    This is vision's original formula, kept byte for byte: mean absolute
    difference, not divided by the magnitude of anything.  Swapping in a
    "cleverer" normalisation would silently re-tune INDUCTION_TAU against a
    different scale, and the number 0.85 was chosen against this one.
    """
    if not descriptor or not simulation or len(descriptor) != len(simulation):
        return 0.0
    return 1.0 - sum(abs(x - y) for x, y in zip(descriptor, simulation)) / len(descriptor)


def induct(descriptor, ledger, next_key=0, tau=INDUCTION_TAU,
           alpha=INDUCTION_ALPHA, first_seen=None):
    """
    File one descriptor into `ledger`, returning (key, next_key).

    `ledger` maps key -> {"simulation": [...], "times_seen": int, ...} and is
    mutated in place - the caller owns the file it lives in.  `next_key` is
    the caller's counter, returned because Python ints are not mutable: the
    caller writes it back where it keeps it.  `first_seen` is an ISO
    timestamp the caller stamps, because the clock belongs to the caller.

    The returned key is an internal index and nothing else: it holds no
    content, is never put into `what`, and means nothing outside this ledger
    (OPEN-21 C).
    """
    best_key = None
    best_sim = 0.0
    for key, entry in ledger.items():
        simulation = entry.get("simulation") or []
        if len(simulation) != len(descriptor):
            continue
        sim = similarity(descriptor, simulation)
        if sim > best_sim:
            best_sim = sim
            best_key = key

    if best_key is not None and best_sim >= tau:
        entry = ledger[best_key]
        simulation = entry.get("simulation") or descriptor
        entry["simulation"] = [
            (1.0 - alpha) * old + alpha * new
            for old, new in zip(simulation, descriptor)
        ]
        entry["times_seen"] = entry.get("times_seen", 0) + 1
        entry["memory_strength"] = min(
            1.0, entry.get("memory_strength", 0.5) + 0.05)
        return best_key, next_key

    key = str(next_key)
    while key in ledger:
        next_key += 1
        key = str(next_key)
    next_key += 1
    ledger[key] = {
        "first_seen": first_seen,
        "times_seen": 1,
        "simulation": list(descriptor),
        "memory_strength": 0.5,
    }
    return key, next_key
