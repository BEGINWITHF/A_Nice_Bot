"""
Tests for PureLearningSystem after DATA-7 read A.

There is no conversation in here any more, and that is the point: the old
demo fed it ten English sentences and then asserted that a vocabulary grew.
A person does not keep a word list in memory, so the system does not keep
one either, and what remains to be tested is the part a person does have -
sequential pattern learning - plus the guarantee that nothing keyed by a
word is left anywhere, on the object or on disk.

pytest runs the demo against a temporary directory and asserts on the result.
Running this file directly still prints the demo into data/test_pure.
"""

import sys
import os
import json

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pure_learning import PureLearningSystem

# Ten short runs of "what came next", as ledger keys perception hands down.
# They are integers on purpose: a pattern is a transition between two
# things, and a person's sequence of things is not made of words.
TEST_SEQUENCES = [
    [0, 3, 3],
    [1, 0, 7],
    [2, 5, 1],
    [0, 3, 9],
    [4, 4, 2],
]

# Everything DATA-7 read A forbids from ever existing.
FORBIDDEN_ON_THE_OBJECT = (
    "word_to_index", "index_to_word", "word_frequency",
    "vocabulary_size", "concepts",
    "meaning_network", "prediction_network",
)

# ...and everything it forbids from ever being written to disk.
FORBIDDEN_ON_DISK = (
    "word_to_index", "index_to_word", "word_frequency",
    "vocabulary_size", "concepts",
)


def run_demo(data_dir):
    """Learn five sequences. Returns the PureLearningSystem."""
    print("Testing Pure Learning System...")
    print("=" * 50)

    ai = PureLearningSystem(data_dir=data_dir)

    print("Initial state - knows nothing")
    print(f"Patterns: {len(ai.patterns)}")
    print()

    for i, sequence in enumerate(TEST_SEQUENCES):
        ai.learn_pattern(sequence)
        print(f"Sequence {i+1}: {sequence}")
        print(f"Patterns so far: {len(ai.patterns)}")
        print()

    print("=" * 50)
    print("Final Statistics:")
    stats = ai.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print()
    print("Test completed successfully!")
    return ai


def test_pure_ai(tmp_path):
    """The system learns transitions, keeps no language, and persists."""
    ai = run_demo(str(tmp_path / "pure"))

    # every adjacent pair of each sequence is one transition, duplicates met
    # again rather than duplicated
    expected = set()
    for sequence in TEST_SEQUENCES:
        for i in range(len(sequence) - 1):
            expected.add((sequence[i], sequence[i + 1]))
    assert len(ai.patterns) == len(expected)

    # DATA-5: a pattern carries a clock and nothing else
    for p in ai.patterns:
        assert set(p) == {"from", "to", "learned_at"}

    # DATA-7 read A: the word structures must not exist at all. This is a
    # regression guard, not a preference - the next person to add a
    # vocabulary back fails here.
    for name in FORBIDDEN_ON_THE_OBJECT:
        assert not hasattr(ai, name), f"{name} came back"

    stats = ai.get_stats()
    assert stats["patterns_learned"] == len(ai.patterns)

    # ...and nothing keyed by a word may reach the disk either
    state_file = os.path.join(str(tmp_path / "pure"), "learning_state.json")
    with open(state_file, "r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    for name in FORBIDDEN_ON_DISK:
        assert name not in on_disk, f"{name} was written back"

    # all of it survives a restart
    again = PureLearningSystem(str(tmp_path / "pure"))
    assert len(again.patterns) == len(ai.patterns)
    for name in FORBIDDEN_ON_THE_OBJECT:
        assert not hasattr(again, name), f"{name} came back after a reload"


if __name__ == "__main__":
    run_demo("data/test_pure")
