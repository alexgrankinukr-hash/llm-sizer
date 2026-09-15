from export import guardrail_problems, pick_reference, REF_4BIT


def idx(models):
    return {"models": models}


def entry(mid, featured=True, min_gb=10.0):
    return {"id": mid, "featured": featured, "sizes": {"min_gb": min_gb}}


def detail(quants, source="config"):
    return {"quants": [{"repo": r, "label": l, "size_bytes": s} for r, l, s in quants], "architecture": {"source": source}}


def test_first_export_has_no_problems():
    assert guardrail_problems(None, {}, idx([entry("a")]), {"a": detail([])}) == []


def test_featured_model_losing_quants_blocks():
    live = idx([entry("a")]); staged = idx([entry("a", min_gb=None)])
    probs = guardrail_problems(live, {"a": detail([("r", "Q4", 10)])}, staged, {"a": detail([])})
    assert any("no quants" in p for p in probs)


def test_size_shift_over_ten_percent_blocks_but_small_shift_passes():
    live = {"a": detail([("r", "Q4_K_M", 100)])}
    ok = guardrail_problems(idx([entry("a")]), live, idx([entry("a")]), {"a": detail([("r", "Q4_K_M", 105)])})
    bad = guardrail_problems(idx([entry("a")]), live, idx([entry("a")]), {"a": detail([("r", "Q4_K_M", 120)])})
    assert ok == [] and any("size moved" in p for p in bad)


def test_new_smaller_quant_is_fine_and_many_vanished_is_not():
    live = {"a": detail([("r", "Q4", 100), ("r", "Q5", 120), ("r", "Q6", 140), ("r", "Q8", 160)])}
    added = guardrail_problems(idx([entry("a")]), live, idx([entry("a")]), {"a": detail([("r", "IQ2", 50), ("r", "Q4", 100), ("r", "Q5", 120), ("r", "Q6", 140), ("r", "Q8", 160)])})
    gone = guardrail_problems(idx([entry("a")]), live, idx([entry("a")]), {"a": detail([("r", "Q4", 100)])})
    assert added == [] and any("vanished" in p for p in gone)


def test_shrinking_index_and_arch_regression_block():
    live_models = [entry(f"m{i}", featured=False) for i in range(20)] + [entry("f")]
    staged_models = [entry(f"m{i}", featured=False) for i in range(10)] + [entry("f")]
    probs = guardrail_problems(idx(live_models), {"f": detail([], "config")}, idx(staged_models), {"f": detail([], "assumed")})
    assert any("shrank" in p for p in probs) and any("regressed" in p for p in probs)


def test_reference_quant_prefers_canonical_labels_and_quantizers():
    qs = [{"label": "UD-Q4_K_XL", "quantizer": "unsloth", "size_gb": 17.6}, {"label": "Q4_K_M", "quantizer": "bartowski", "size_gb": 17.8},
          {"label": "Q4_K_M", "quantizer": "lmstudio-community", "size_gb": 16.8}, {"label": "MLX-4bit", "quantizer": "mlx-community", "size_gb": 16.0}]
    assert pick_reference(qs, REF_4BIT)["size_gb"] == 16.8
    assert pick_reference([qs[3]], REF_4BIT)["label"] == "MLX-4bit"
    assert pick_reference([], REF_4BIT) is None
