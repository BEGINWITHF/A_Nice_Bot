"""
Look inside the bot's mind - read only.

Prints what the memory pipeline is currently holding, what each long term
store remembers, and the sleep ledger. Nothing is written, no camera and no
microphone is touched, so it is safe to run while the bot is sleeping or
while main.py is running.

    python memory_report.py
"""

import json
import os
import time
from datetime import datetime

from core import parameters as P
from core.memory_pipeline import DEFAULT_POLICY
from core.parameters import accessibility

DATA = "data"


def load(path, default=None):
    """Read a state file, or hand back `default` if there is nothing yet."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError):
        return default


def head(title):
    print()
    print(title)
    print("-" * len(title))


def when(ts):
    """Unix time -> HH:MM:SS, so reports are readable at a glance."""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "?"


def age_of(ts):
    seconds = max(0.0, time.time() - float(ts))
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def entry_line(entry, load):
    """One line of a short term entry: salience, availability, hits, content."""
    payload = json.dumps(entry.get("payload", {}), ensure_ascii=False)
    if len(payload) > 60:
        payload = payload[:57] + "..."
    # Nothing is stored: availability is computed from the rehearsal clock
    # every time it is asked for (DATA-4 / DATA-5).
    acc = accessibility(max(0.0, time.time() - entry.get("when_last_rehearsed",
                                                        time.time())), load)
    return (f"  {entry.get('salience', 0):.2f}  "
            f"acc {acc:.2f}  "
            f"x{entry.get('hits', 1)}  "
            f"{entry.get('kind', '?'):<8} "
            f"{age_of(entry.get('when_last_rehearsed', time.time()))}  "
            f"{payload}")


def show_pipeline():
    state = load(f"{DATA}/memory/pipeline_state.json", {})
    if not state:
        print("\nNo pipeline state yet - run `python main.py` first.")
        return

    focus = state.get("consciousness", [])
    recent = state.get("recent", [])
    # Ledgers written before OPEN-15 called the lower layer `short_term`.
    recent = recent + [e for e in state.get("short_term", []) if isinstance(e, dict)]

    focus_cap = DEFAULT_POLICY["conscious_capacity"]
    recent_cap = DEFAULT_POLICY["recent_capacity"]
    crowding = len(recent) / float(recent_cap)

    head(f"FOCUS   ({len(focus)}/{focus_cap} - what is in mind right now)")
    for entry in focus:
        print(entry_line(entry, crowding))
    if not focus:
        print("  (empty)")

    head(f"RECENT   ({len(recent)} entries, soft capacity {recent_cap}, "
         f"load {crowding:.2f})")
    print("soft: crowding makes everything older faster, nothing is evicted")
    if not recent:
        print("  (empty - consolidated or forgotten at the last sleep)")
    for entry in recent:
        print(entry_line(entry, crowding))

    show_episodes()

    show_rhythm(state)

    head("COUNTERS")
    print(f"memorised {state.get('total_recorded', 0)}   "
          f"rejected {state.get('total_dropped', 0)}   "
          f"ignored while asleep {state.get('total_dropped_asleep', 0)}   "
          f"{state.get('events_since_sleep', 0)} events since last pass   "
          f"gate: salience >= {DEFAULT_POLICY['min_salience']}")
    print(f"startled awake {state.get('startle_count', 0)} times "
          f"(salience >= {DEFAULT_POLICY['startle_salience']}, "
          f"then {DEFAULT_POLICY['arousal_s'] // 60} min alert)")


def show_rhythm(state):
    """OPEN-12: where the two processes are right now - computed, never stored.

    Nothing in this block is a schedule: the state is whatever Process S is
    doing against a circadian-modulated pair of thresholds.
    """
    head("RHYTHM   (two-process model, OPEN-12)")
    now = time.time()
    asleep = state.get("asleep", False)
    pressure = float(state.get("pressure", P.H_SLEEP))
    upper, lower = P.thresholds(now)
    since = state.get("state_since", now)

    print(f"  {'ASLEEP ' if asleep else 'AWAKE  '}"
          f"{age_of(since)}   pressure {pressure:.3f}")
    if asleep:
        verdict = ("waking is due" if pressure <= lower
                   else "sleeping it off - senses still on, nothing memorised")
    elif pressure >= upper:
        verdict = "at or above the upper threshold -> sleep is due"
    elif pressure <= lower:
        verdict = "wide awake - pressure fully spent, nothing can wake it"
    else:
        verdict = ("between the thresholds -> both states are possible "
                   "(this is where a startle can hold)")
    print(f"  thresholds now: sleep >= {upper:.3f}, wake <= {lower:.3f}")
    print(f"  {verdict}")
    print(f"  circadian {P.circadian_signal(now):+.3f} "
          f"(+1 at 18:00, -1 at the "
          f"{time.strftime('%H:%M', time.gmtime(P.CIRCADIAN_MIN_PHASE_S))} "
          f"trough)   sort every "
          f"{DEFAULT_POLICY['sleep_cycle'] // 60} min while asleep")


def show_episodes():
    """The long term episode store (OPEN-17): five fields, curve forgotten."""
    head("EPISODES   (long term: what sleep decided to keep)")
    data = load(f"{DATA}/memory/episodes.json", {})
    episodes = (data or {}).get("episodes", [])
    if not episodes:
        print("  nothing transferred yet "
              "(needs a replay AND >= 1 day of age at a sleep)")
        return
    budget = P.LONG_TERM_BUDGET
    print(f"{len(episodes)}/{budget} episodes, "
          f"curve floor {DEFAULT_POLICY['forget_threshold']}, "
          f"replay salience >= {DEFAULT_POLICY['replay_salience']}")
    ranked = sorted(episodes, key=lambda e: e.get("when_last_rehearsed", 0),
                    reverse=True)
    for ep in ranked[:12]:
        what = json.dumps(ep.get("what"), ensure_ascii=False, default=str)
        if len(what) > 56:
            what = what[:53] + "..."
        acc = accessibility(max(0.0, time.time() - ep.get("when_last_rehearsed",
                                                          time.time())),
                            len(episodes) / float(budget))
        print(f"  acc {acc:.2f}  first {age_of(ep.get('when_first', time.time()))}  "
              f"last {age_of(ep.get('when_last_rehearsed', time.time()))}  "
              f"{ep.get('kind', '?'):<8} {what}")


def show_seeing():
    state = load(f"{DATA}/senses/seeing/seeing_state.json")
    head("SEEING   (long term: what it has ever seen)")
    if not state:
        print("  no state yet")
        return
    print(f"visual acuity {state.get('visual_acuity', 0):.3f}  "
          f"(grows with use)")

    known = state.get("known_objects", {})
    if not known:
        print("  nothing recognised yet")
        return
    ranked = sorted(known.items(),
                    key=lambda kv: kv[1].get("times_seen", 0), reverse=True)
    print(f"{len(ranked)} distinct objects, strongest first:")
    for name, data in ranked[:12]:
        first = str(data.get("first_seen", ""))[:19].replace("T", " ")
        print(f"  {name:<24} seen {data.get('times_seen', 0):>4}x  "
              f"strength {data.get('memory_strength', 0):.2f}  "
              f"since {first}")


def show_hearing():
    state = load(f"{DATA}/senses/hearing/hearing_state.json")
    head("HEARING   (long term: what it has ever heard)")
    if not state:
        print("  no state yet")
        return
    print(f"sensitivity {state.get('hearing_sensitivity', 0):.3f}  "
          f"({len(state.get('words_recognized', []))} words recognised, "
          f"{len(state.get('sound_frequencies', {}))} frequencies stored)")

    memory = state.get("auditory_memory", {})
    if not memory:
        print("  no sounds memorised yet")
        return
    ranked = sorted(memory.items(),
                    key=lambda kv: kv[1] if isinstance(kv[1], (int, float))
                    else kv[1].get("times_heard", 0) if isinstance(kv[1], dict)
                    else 0, reverse=True)
    for name, data in ranked[:12]:
        if isinstance(data, dict):
            times = data.get("times_heard", data.get("count", 1))
            strength = data.get("memory_strength", data.get("strength", 0))
            print(f"  {name:<24} heard {times:>4}x  strength {strength:.2f}")
        else:
            print(f"  {name:<24} {data}")


def show_human():
    state = load(f"{DATA}/pure/human/human_state.json")
    head("HUMAN   (state: mood, personality, preferences - holds no memories)")
    if not state:
        print("  no state yet")
        return
    context = state.get("social_context", {})
    print(f"mood {state.get('mood', '?')}  "
          f"mood energy {state.get('mood_energy', 0):.2f}  "
          f"energy {state.get('energy_level', 0):.2f}  "
          f"trust {context.get('trust_level', 0)}")
    print(f"{context.get('conversation_count', 0)} interactions")
    # OPEN-19: this store used to keep its own `interaction_memories` /
    # `long_term_memories` alongside the pipeline's three layers.  The author
    # had it cut (DATA-5, IO-2, OPEN-10, OPEN-15), so a file that still shows
    # them has simply not been rewritten yet - the next save drops them.
    leftovers = [k for k in ("interaction_memories", "long_term_memories")
                 if state.get(k)]
    if leftovers:
        print(f"  legacy keys awaiting the next save: {', '.join(leftovers)}")


def show_learning():
    state = load(f"{DATA}/pure/learning_state.json")
    head("LEARNING   (long term: patterns learned)")
    if not state:
        print("  no state yet")
        return
    print(f"{len(state.get('patterns', []))} learned patterns")

    # An older ledger still carrying a vocabulary is reported as what it is:
    # language waiting to be erased on the next save, not knowledge.
    leftovers = [k for k in ("word_to_index", "index_to_word",
                             "word_frequency", "vocabulary_size", "concepts")
                 if state.get(k)]
    if leftovers:
        print(f"  legacy keys awaiting the next save: {', '.join(leftovers)}")


def show_sensory():
    state = load(f"{DATA}/senses/sensory_state.json")
    head("SENSORY   (body state)")
    if not state:
        print("  no state yet")
        return
    print(f"awareness {state.get('awareness_level', 0):.2f}  "
          f"load {state.get('sensory_load', 0):.2f}  "
          f"fatigue {state.get('fatigue_level', 0):.2f}")


def show_ledger():
    state = load(f"{DATA}/memory/pipeline_state.json", {})
    reports = state.get("sleep_reports", [])
    head(f"SLEEP LEDGER   ({len(reports)} sleeps, newest last)")
    if not reports:
        print("  no sleep has happened yet")
        return
    for report in reports[-8:]:
        print(f"  {report.get('started', '?')[:19].replace('T', ' ')}  "
              f"{report.get('reason', '?'):<9} "
              f"{report.get('state', '?'):<7} "
              f"{(report.get('transition') or '-'):<11} "
              f"awake {report.get('awake_seconds', 0):>6.1f}s  "
              f"events {report.get('events_processed', 0):>3}  "
              f"focus {report.get('focus_in', 0):>2}  "
              f"recent {report.get('recent_in', 0):>3}->{report.get('recent_out', 0):<3} "
              f"replayed {report.get('replayed', 0):>2}  "
              f"to LT {len(report.get('promoted', [])):>2}  "
              f"dropped {report.get('dropped_below_threshold', 0):>2}  "
              f"episodes {report.get('episodes', 0):>3}")
        stores = report.get("stores", {})
        if stores:
            parts = []
            for name, info in stores.items():
                if isinstance(info, dict) and "forgotten" in info:
                    # `forgotten` is an int for simple stores and a dict of
                    # field -> count for the ones that hold several lists.
                    forgotten = info["forgotten"]
                    if isinstance(forgotten, dict):
                        gone = sum(v for v in forgotten.values()
                                   if isinstance(v, (int, float)))
                    else:
                        gone = forgotten or 0
                    consolidated = info.get("consolidated", 0)
                    if isinstance(consolidated, dict):
                        consolidated = sum(v for v in consolidated.values()
                                           if isinstance(v, (int, float)))
                    parts.append(f"{name}: kept {consolidated} / "
                                 f"forgot {gone}")
            if parts:
                print("      " + "   ".join(parts))


def show_files():
    head("FILES   (the mind on disk - data/ is not tracked by git)")
    rows = []
    for base, _, names in os.walk(DATA):
        for name in names:
            path = os.path.join(base, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            rows.append((path.replace("\\", "/"), stat.st_size, stat.st_mtime))
    if not rows:
        print("  data/ is empty")
        return
    for path, size, mtime in sorted(rows):
        print(f"  {path:<46} {size:>8} B  {when(mtime)}")


def main():
    print("=" * 68)
    print("A_Nice_Bot - what the mind currently holds")
    print("=" * 68)
    show_pipeline()
    show_seeing()
    show_hearing()
    show_human()
    show_learning()
    show_sensory()
    show_ledger()
    show_files()
    print()


if __name__ == "__main__":
    main()
