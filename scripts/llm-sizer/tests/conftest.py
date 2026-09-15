import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

FIXTURES = HERE / "fixtures"


@pytest.fixture
def configs() -> dict:
    return {p.stem.replace(".config", ""): json.loads(p.read_text()) for p in (FIXTURES / "configs").glob("*.config.json")}


@pytest.fixture
def repos() -> dict:
    return {p.stem.replace("__", "/"): json.loads(p.read_text()) for p in (FIXTURES / "repos").glob("*.json")}


@pytest.fixture
def listing() -> list:
    return json.loads((FIXTURES / "listing.json").read_text())
