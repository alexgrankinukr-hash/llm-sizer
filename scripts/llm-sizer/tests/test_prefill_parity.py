"""The prefill parity fixture pins speed_model.py's prefill mirror (and, through speed-parity.test.ts, the engine)."""
import json

from common import HERE, PUBLIC_DATA_DIR
from speed_model import first_word_wait_s, prefill_constant, prefill_d0, prefill_rate_at, prefill_tok_s

CASES = json.loads((HERE / "tests" / "fixtures" / "prefill-cases.json").read_text())["cases"]
PF = json.loads((PUBLIC_DATA_DIR / "factors.json").read_text())["prefill"]


def test_cases_cover_sources_runtimes_and_classes():
    assert {c["expected_source"] for c in CASES} >= {"measured", "generation", "assumed"}
    assert {c["runtime"] for c in CASES} == {"gguf", "mlx"}
    assert len({c["arch_class"] for c in CASES}) >= 3


def test_every_case_reproduces():
    for c in CASES:
        k, source, _ = prefill_constant(PF, c["chip"], c["gpu_cores"], c["platform"])
        assert source == c["expected_source"], c["id"]
        assert abs(k - c["expected_k"]) < 1e-6, c["id"]
        pf = dict(PF); pf["_mlx_group"] = c["mlx_group"]
        tok = prefill_tok_s(pf, k, c["quant_label"], c["format"], c["arch_class"], c["runtime"], c["active_params_b"])
        assert abs(tok - c["expected_tok_s"]) < 0.002, c["id"]
        d0 = prefill_d0(PF, c["kv_layers"], c["layers"])
        assert abs(d0 - c["expected_d0"]) < 0.002, c["id"]
        assert abs(first_word_wait_s(tok, d0, 8192) - c["expected_wait_8k_s"]) < 1e-3, c["id"]
        assert abs(prefill_rate_at(tok, d0, c["context_tokens"]) - c["expected_rate_at_ctx"]) < 1e-3, c["id"]


def test_the_physics_holds():
    """Prefill does not care about the file's bits, doubles with the cores, and a 32K prompt takes more than four 8K ones."""
    k, _, _ = prefill_constant(PF, "M5 Max", 40, "apple")
    q4 = prefill_tok_s(PF, k, "Q4_0", "gguf", "dense", "gguf", 27.0)
    q8 = prefill_tok_s(PF, k, "Q8_0", "gguf", "dense", "gguf", 27.0)
    assert q4 == q8
    k32, _, _ = prefill_constant(PF, "M5 Max", 32, "apple")
    assert abs(k32 / k - 0.8) < 1e-9
    d0 = prefill_d0(PF, 16, 64)
    assert first_word_wait_s(q4, d0, 32768) > 4 * first_word_wait_s(q4, d0, 8192)
