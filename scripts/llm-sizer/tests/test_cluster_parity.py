"""The cluster parity fixture pins speed_model.py's linked-machine mirror (and, through the engine's own
cluster-parity test, src/lib/llm-sizer/engine/cluster.ts) to the fitted numbers in factors.json."""
import json

from common import HERE, PUBLIC_DATA_DIR
from speed_model import cluster_ms, link_class, prefill_cluster_factor

CASES = json.loads((HERE / "tests" / "fixtures" / "cluster-cases.json").read_text())["cases"]
CF = json.loads((PUBLIC_DATA_DIR / "factors.json").read_text())["cluster"]


def test_cases_cover_every_link_split_kind_and_count():
    assert {c["link"] for c in CASES} == {"mac", "spark", "gpu"}
    assert {c["split"] for c in CASES} == {"layer", "tensor"}
    assert {c["kind"] for c in CASES} == {"dense", "moe"}
    assert {c["n"] for c in CASES} == {2, 3, 4}
    assert {c["single_ms"] for c in CASES} == {20.0, 50.0, 80.0}
    assert len(CASES) == 36 and len({c["id"] for c in CASES}) == len(CASES)


def test_every_case_reproduces():
    for c in CASES:
        ms = cluster_ms(c["single_ms"], c["n"], c["split"], c["link"], c["kind"], CF)
        assert abs(ms - c["expected_ms"]) < 1e-6, (c["id"], ms, c["expected_ms"])
        assert c["expected_tok_s"] == 1000 / c["expected_ms"], c["id"]
        assert abs(prefill_cluster_factor(c["n"], c["split"], CF) - c["prefill_factor"]) < 1e-9, c["id"]


def test_the_shape_of_the_two_formulas():
    """A layer split adds a hop per extra machine and never divides; tensor parallel divides and pays log2."""
    one = 50.0
    layer = [cluster_ms(one, n, "layer", "mac", "dense", CF) for n in (1, 2, 3, 4)]
    assert layer[0] == one and all(b > a for a, b in zip(layer, layer[1:]))
    assert abs((layer[3] - layer[0]) / 3 - CF["hop_ms"]["mac"]["value"]) < 1e-9
    tensor = [cluster_ms(one, n, "tensor", "mac", "dense", CF) for n in (1, 2, 4)]
    assert tensor[0] == one and tensor[1] < one and tensor[2] < tensor[1]
    assert abs(tensor[2] - (one / 4 + 2 * CF["tensor_c_ms"]["dense"]["value"])) < 1e-9
    # a mixture of experts pays the larger constant, and cards in one box the assumed small one
    assert cluster_ms(one, 4, "tensor", "mac", "moe", CF) > cluster_ms(one, 4, "tensor", "mac", "dense", CF)
    assert cluster_ms(one, 4, "tensor", "gpu", "moe", CF) == cluster_ms(one, 4, "tensor", "gpu", "dense", CF)
    # one machine is one machine on either split
    for split in ("layer", "tensor"):
        assert cluster_ms(one, 1, split, "spark", "moe", CF) == one
        assert prefill_cluster_factor(1, split, CF) == 1.0


def test_link_classes():
    assert link_class("M5 Max", "apple") == "mac" and link_class("M3 Ultra", "apple") == "mac"
    assert link_class("DGX Spark", "cuda") == "spark" and link_class("GB10", "cuda") == "spark"
    assert link_class("RTX PRO 6000 Blackwell", "cuda") == "gpu" and link_class("RX 7900 XTX", "rocm") == "gpu"
