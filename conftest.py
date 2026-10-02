"""
Fixtures the whole suite shares.

One rule, enforced once here instead of once per test file: a test run must
touch no hardware.  There used to be a second rule about the network, added
with OPEN-10 so that a green build would not depend on github.com being up
while the detector fetched its weights.  The detector went with OPEN-10 a
- perception is the bot's own now - and with it went the only thing in the
repository that reached for the network.
"""

import pytest

from core.seeing import SeeingSystem


@pytest.fixture(autouse=True)
def no_camera(monkeypatch):
    """These tests must never touch hardware."""
    monkeypatch.setattr(SeeingSystem, "_check_camera", lambda self: None)
