import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from playground.provider import SyntheticProvider


@pytest.fixture(scope="session")
def syn():
    return SyntheticProvider(start="2016-01-04", end="2017-12-29")


@pytest.fixture(scope="session")
def syn_events(syn):
    return syn.synthetic_events()
