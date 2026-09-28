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
| `core/sensory.py` | Camera + microphone front end, body state (awareness, load) |
| `core/seeing.py` / `core/hearing.py` | One store per sense, each with real forgetting |
| `core/pure_learning.py` | Vocabulary, word patterns and concepts learned from experience |
| `core/human_like.py` | Mood, trust, relationships, long-term life events |
| `core/baby_brain.py` | Developmental stages (birth → sensory → babbling → …) |
| `core/pure_network.py` | The network itself, written in plain Python |
| `configs/` | Paths only — no external model is configured anywhere |
| `ui/`, `gui_main.py` | Experimental GUI, will be rebuilt |
| `tools/` | Dormant device-control helpers, not wired into the loop yet |
| `test_*.py` | Test suite |

## Requirements

- Python 3.11 or newer
- `pip install -r requirements.txt` → `numpy`, `opencv-python`, `pyaudio`
- A camera and a microphone (tests do not need either)

Optional, only if you use them:

```text
pip install PySide6   # GUI: python gui_main.py
pip install pynput mss # tools/: keyboard, mouse, screenshot helpers
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
  slept: replayed=2 dropped=0 short_term=7
```

Press `Ctrl+C` to stop. The bot sleeps once more before exiting, so nothing is
lost.

## Look inside

```bash
python memory_report.py
```

```text
SHORT-TERM BUFFER   (what it is holding right now)
--------------------------------------------------
1/128 entries, 0 events since last sleep
gate: salience >= 0.25, sleep after 900s or 200 events
  0.30  str 0.84  x8  seeing   25m ago  {"objects": ["red1_object"]}

SEEING   (long term: what it has ever seen)
  red1_object              seen   30x  strength 1.00  since 2026-09-29 01:51

SLEEP LEDGER   (3 sleeps, newest last)
  2026-09-29 02:16:19  shutdown   awake   24.2s  events  7  promoted 0 ...
```

It only reads state files: no camera, no microphone, nothing is written, and it
is safe to run while `main.py` is running.

## Tests

```bash
python -m pytest -v
```

28 tests in about three seconds. They need **only pytest** — `core/` imports
nothing outside the standard library at module level, so no camera, no
microphone, no model weights, and no writes outside a temporary directory.

The same suite runs on every push through GitHub Actions
(`.github/workflows/tests.yml`).

## State

`data/` holds the whole mind: the short-term buffer, the sleep ledger and every
long-term store. It is listed in `.gitignore` and never committed. Deleting it
only resets the bot, it does not affect the code.

```bash
python clean_memories.py   # erase everything, start fresh
python memory_report.py    # verify data/ is empty
```

## Status

This is a long-running project, built one complete step at a time.

- **Done:** sensory loop, salience gate, short-term buffer, sleep with
  consolidation and real forgetting across all five stores, persistence across
  restarts, test suite with CI.
- **Next:** capability part — an HTTP ingestion endpoint so other devices can
  feed the same senses in.

## License

GPL-3.0 — see [LICENSE](LICENSE).
