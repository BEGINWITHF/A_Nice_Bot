# A_Nice_Bot

An embodied agent that learns the world the way a baby does: through a camera
and a microphone. **No text input, no text output, no pre-trained chat model.**
It senses the room, keeps what is worth keeping, forgets what is not, and
sleeps on a rhythm to consolidate what it learned.

Everything the agent holds counts as one stream of consciousness, so runtime
state is never committed to this repository — pictures are not memories.

## Layout

| Path | What it is |
| --- | --- |
| `main.py` | The main loop: see → hear → memorise → sleep when due |
| `memory_report.py` | **Read-only view** of everything the bot currently holds |
| `clean_memories.py` | Wipe all state and start from a blank mind |
| `core/memory_pipeline.py` | The gate: when to memorise, when to forget, when to sleep |
| `core/parameters.py` | Every number the memory system uses, with its source and how to swap it (`OPEN-13`) |
| `core/episodes.py` | The long-term episode store sleep transfers into (`OPEN-17`) |
| `core/sensory.py` | Camera + microphone front end, body state (awareness, load) |
| `core/seeing.py` / `core/hearing.py` | One store per sense, each with real forgetting |
| `core/pure_learning.py` | Vocabulary, word patterns and concepts learned from experience |
| `core/human_like.py` | Mood, personality, preferences, social state - holds no memories (`OPEN-19`) |
| `core/baby_brain.py` | Developmental stages (birth → sensory → babbling → …) |
| `core/pure_network.py` | The network itself, written in plain Python |
| `ui/`, `gui_main.py` | Experimental GUI, will be rebuilt |
| `test_*.py` | Test suite |

There is no external model to configure anywhere in the repository: the network
lives in `core/pure_network.py` and is plain Python, because the model is one we
train ourselves.

## Requirements

- Python 3.11 or newer
- `pip install -r requirements.txt` → `numpy`, `opencv-python`, `pyaudio`
- A camera and a microphone (tests do not need either)

Optional, only if you use them:

```text
pip install PySide6   # GUI: python gui_main.py
```

## Run

```bash
python main.py
```

```text
PURE AI - Sensory Experience
========================================
Microphone: Ready
Camera: Ready

The AI is experiencing the world...
Press Ctrl+C to stop
========================================
```

Each cycle takes one frame and one second of audio. A **salience gate** decides
whether the observation is worth remembering at all; a frame that shows nothing
new never enters memory. Every 20 cycles a one-line status is printed, and when
the sleep rhythm comes due it prints a sleep report:

```text
  slept: replayed=2 dropped=0 recent=7 episodes=3
```

Press `Ctrl+C` to stop. The bot sleeps once more before exiting, so nothing is
lost.

## Look inside

```bash
python memory_report.py
```

The mind is held in three layers: a **focus** of six slots (what is in mind
right now), a **recent** cache below it whose capacity is soft - crowding it
makes everything in it age faster instead of throwing anything out - and the
**long-term episodes** sleep transfers into. Everything numeric lives in
`core/parameters.py`, next to the paper it came from.

```text
FOCUS   (0/6 - what is in mind right now)
-----------------------------------------
  (empty)

RECENT   (1 entries, soft capacity 65, load 0.02)
-------------------------------------------------
soft: crowding makes everything older faster, nothing is evicted
  0.25  acc 1.00  x14  seeing   0s ago  {"objects": ["red1_object"]}

EPISODES   (long term: what sleep decided to keep)
--------------------------------------------------
  acc 1.00  first 2d ago  last 2d ago  seeing   {"objects": ["red1_object"]}

RHYTHM   (two-process model, OPEN-12)
-------------------------------------
  AWAKE  4h ago   pressure 0.170
  thresholds now: sleep >= 0.693, wake <= 0.193
  wide awake - pressure fully spent, nothing can wake it
  circadian +0.195 (+1 at 18:00, -1 at the 06:00 trough)   sort every 90 min while asleep

SEEING   (long term: what it has ever seen)
  red1_object              seen   30x  strength 1.00  since 2026-09-29 01:51

SLEEP LEDGER   (3 passes, newest last)
  2026-09-30 01:27:04  scheduled asleep  fell_asleep  awake  ... recent 7->6 ...
```

`acc` is availability recomputed from the last-rehearsal clock every time you
ask - no strength field is stored anywhere **in the pipeline's three layers**,
because a stored parameter is not part of a memory (`DATA-4`/`DATA-5`). The
`SEEING` block above still prints one: the sensory ledgers keep theirs until
`IO-6` rewires them, which `OPEN-17` explicitly parked. `human_like` - the one
store that also held `strength`/`importance` outside the senses - lost its
whole memory subsystem instead (`OPEN-19`).

**There is no sleep schedule.** `RHYTHM` shows sleep pressure against a pair
of thresholds that the circadian signal moves up and down; crossing upward is
falling asleep, crossing downward is waking. The shipped constants settle into
one 16 h awake / 8 h asleep day (7.9 / 16.1 measured, exactly 24 h, phase
locked) - a test integrates them for two weeks to prove it. While asleep the
senses still run but ordinary observations are not memorised; a severe one
startles the bot awake.

It only reads state files: no camera, no microphone, nothing is written, and it
is safe to run while `main.py` is running.

## Tests

```bash
python -m pytest -v
```

49 tests in about two seconds. They need **only pytest** - `core/` imports
nothing outside the standard library at module level, so no camera, no
microphone, no model weights, and no writes outside a temporary directory.

The same suite runs on every push through GitHub Actions
(`.github/workflows/tests.yml`).

## State

`data/` holds the whole mind: the focus, the recent cache, the long-term
episodes, the sleep ledger and every store. It is listed in `.gitignore` and
never committed. Deleting it only resets the bot, it does not affect the code.

```bash
python clean_memories.py   # erase everything, start fresh
python memory_report.py    # verify data/ is empty
```

## Status

This is a long-running project, built one complete step at a time.

- **Done:** sensory loop, salience gate, three-layer memory (focus / recent /
  long-term episodes) with a soft capacity, event segmentation by window plus
  content shock, sleep with selective transfer and real forgetting across all
  five stores, a two-process sleep rhythm with no schedule in it, sleeping-but-
  still-sensing behaviour, persistence across restarts, test suite with CI.
- **Next:** the capability part - an HTTP ingestion endpoint so other devices
  can feed the same senses in (`CAP-8`).

## License

GPL-3.0 — see [LICENSE](LICENSE).
