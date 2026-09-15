"""The parity fixture pins speed_model.py (and, through speed-parity.test.ts, the engine) to the fitted numbers."""
import json

from common import HERE, PUBLIC_DATA_DIR
from speed_model import decode_ms, gb_per_token, quant_family

CASES = json.loads((HERE / "tests" / "fixtures" / "speed-cases.json").read_text())["cases"]


def test_cases_cover_both_runtimes_and_several_classes():
    assert len(CASES) >= 6
    assert {c["runtime"] for c in CASES} == {"gguf", "mlx"}
    assert len({c["arch_class"] for c in CASES}) > 1


def test_every_case_reproduces():
    sm = json.loads((PUBLIC_DATA_DIR / "factors.json").read_text())["speed_model"]
    for c in CASES:
        gb = gb_per_token(c["active_params_b"], c["bits"], c["kv_bytes_per_token_8bit"], c["kv_bits"], c["context_tokens"])
        assert abs(gb - c["gb_per_token"]) < 1e-5, c["id"]
        fam = quant_family(c["quant_label"], "mlx" if c["runtime"] == "mlx" else "gguf")
        ms = decode_ms(c["profile"], c["bandwidth_gbs"], gb, fam, c["arch_class"], c["runtime"], c["mlx_group"], c["attention"], c["context_tokens"], sm, c.get("platform", "apple"), c.get("mlx_ctx", "pre_m5"))
        assert abs(ms - c["expected_ms"]) < 0.002, (c["id"], ms, c["expected_ms"])
        assert abs(1000 / ms - c["expected_tok_s"]) < 0.05, c["id"]
