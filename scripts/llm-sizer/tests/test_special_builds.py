import json
from pathlib import Path

import pytest

from export import special_builds


GOOD = {"label": "IQ4_XS · engram on SSD", "resident_gb": 45.8, "ssd_gb": 39.1, "disk_gb": 84.9, "format": "gguf", "bits": 3.84}


def test_passes_a_complete_build_through_unchanged():
    assert special_builds({"special_builds": [GOOD]}, "m") == [GOOD]


def test_no_builds_is_an_empty_list():
    assert special_builds({}, "m") == []
    assert special_builds({"special_builds": None}, "m") == []


@pytest.mark.parametrize("broken, message", [
    ({**GOOD, "label": ""}, "needs a label"),
    ({k: v for k, v in GOOD.items() if k != "ssd_gb"}, "ssd_gb must be"),
    ({**GOOD, "resident_gb": "45.8"}, "resident_gb must be"),
    ({**GOOD, "format": "safetensors"}, "format must be"),
    ({**GOOD, "disk_gb": 10}, "disk_gb must be"),
])
def test_a_broken_build_fails_the_export(broken, message):
    with pytest.raises(ValueError, match=message):
        special_builds({"special_builds": [broken]}, "m")


def test_the_live_overrides_pass_the_check():
    ov = json.loads((Path(__file__).resolve().parents[1] / "models.overrides.json").read_text())
    for model_id, entry in ov["models"].items():
        special_builds(entry, model_id)
