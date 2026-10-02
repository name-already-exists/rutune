import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synth import SR

from app.config import Config


@pytest.fixture
def cfg() -> Config:
    c = Config()
    c.audio.sample_rate = SR
    c.world.f0_method = "dio"  # faster than harvest; good enough on clean synthetic audio
    return c
