"""
Fixtures the whole suite shares.

Two rules, enforced once here instead of once per test file: a test run must
touch no hardware and reach no network.  The hardware half is old.  The
network half is new with OPEN-10 - the detector fetches its weights the first
time a frame is seen, which is right in production and wrong in a test run,
where a green build must not depend on github.com being up.
"""

import pytest

from core import vision_model
from core.seeing import SeeingSystem


@pytest.fixture(autouse=True)
def no_camera(monkeypatch):
    """These tests must never touch hardware."""
    monkeypatch.setattr(SeeingSystem, "_check_camera", lambda self: None)


@pytest.fixture(autouse=True)
def no_model_fetch(monkeypatch):
    """
    The detector must not download anything while tests are running.

    Every test that wants the real fetch back can ask for it by patching
    ``vision_model.fetch_model`` again with the function it captured at import
    time; the teardown below unwinds both.
    """

    def blocked(path=None):
        raise RuntimeError("vision model fetch is disabled in tests")

    monkeypatch.setattr(vision_model, "fetch_model", blocked)
