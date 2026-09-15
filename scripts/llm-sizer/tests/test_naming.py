import naming as n


def test_strip_suffix_kinds():
    assert n.strip_suffix("Qwen3.8-27B-GGUF").kind == "gguf"
    s = n.strip_suffix("Qwen3.8-27B-MLX-4bit")
    assert (s.kind, s.bits, s.label, s.stem) == ("mlx", 4.0, "MLX-4bit", "Qwen3.8-27B")
    assert n.strip_suffix("DeepSeek-V4-Flash-2bit").label == "MLX-2bit"
    q = n.strip_suffix("gemma-4-26b-a4b-it-qat-q4_0-gguf")
    assert q.kind == "gguf" and q.qat and q.stem == "gemma-4-26b-a4b-it"
    p = n.strip_suffix("GLM-5.2-FP8")
    assert (p.kind, p.bits, p.label) == ("precision", 8.0, "FP8")
    assert n.strip_suffix("DeepSeek-V4-Flash").kind == "none"


def test_stem_key_and_variant():
    assert n.stem_key("gemma-4-26B-A4B-it") == "gemma-4-26b-a4b"
    assert n.same_model("Mistral-Small-4-Instruct-2606", "mistral-small-4-2606")
    assert not n.same_model("DeepSeek-V4-Flash-0731", "DeepSeek-V4-Flash")
    assert n.variant("gemma-4-26b-a4b-pt") == "base"
    assert n.variant("Qwen3-30B-Thinking") == "thinking"
    assert n.variant("Qwen3.8-27B") == "instruct"
    assert n.slugify("GLM-5.3-Flash") == "glm-5.3-flash"
    assert n.slugify("Qwen3.8-27B") == "qwen3.8-27b"


def test_classify_accepts_real_quantizer_names():
    cases = {
        "unsloth/Qwen3.8-27B-GGUF": ("gguf", "unsloth"),
        "bartowski/Qwen3.8-27B-GGUF": ("gguf", "bartowski"),
        "lmstudio-community/Qwen3.8-27B-MLX-4bit": ("mlx", "lmstudio-community"),
        "mlx-community/Qwen3.8-27B-8bit": ("mlx", "mlx-community"),
        "Qwen/Qwen3.8-27B-FP8": ("safetensors", "official"),
        "ggml-org/Qwen3.8-27B-GGUF": ("gguf", "ggml-org"),
    }
    for repo, (fmt, quantizer) in cases.items():
        m = n.classify_quant_repo(repo, "Qwen", "Qwen3.8-27B")
        assert m.accepted, (repo, m.reject_reason)
        assert (m.format, m.quantizer) == (fmt, quantizer), repo


def test_classify_org_prefixes():
    m = n.classify_quant_repo("bartowski/google_gemma-4-26B-A4B-it-GGUF", "google", "gemma-4-26b-a4b")
    assert m.accepted and m.format == "gguf" and m.variant == "instruct"
    m = n.classify_quant_repo("bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF", "Qwen", "Qwen3.8-27B")
    assert not m.accepted and m.reject_reason.startswith("variant_token")
    m = n.classify_quant_repo("bartowski/someone_Qwen3.8-27B-GGUF", "Qwen", "Qwen3.8-27B")
    assert not m.accepted and m.reject_reason == "foreign_prefix:someone"
    m = n.classify_quant_repo("mlx-community/deepseek-ai-DeepSeek-V4-Flash-8bit", "deepseek-ai", "DeepSeek-V4-Flash")
    assert m.accepted and m.label == "MLX-8bit"
    m = n.classify_quant_repo("mlx-community/cyberneurova-Qwen3.8-27B-8bit", "Qwen", "Qwen3.8-27B")
    assert not m.accepted and m.reject_reason.startswith("foreign_prefix")


def test_classify_rejects_variants_and_unknowns():
    assert not n.classify_quant_repo("mlx-community/DeepSeek-V4-Flash-2bit-DQ", "deepseek-ai", "DeepSeek-V4-Flash").accepted
    assert not n.classify_quant_repo("mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit", "deepseek-ai", "DeepSeek-V4-Flash").accepted
    assert not n.classify_quant_repo("mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed", "deepseek-ai", "DeepSeek-V4-Flash").accepted
    assert not n.classify_quant_repo("mlx-community/Qwen3.8-27B-8bit-MTP", "Qwen", "Qwen3.8-27B").accepted
    m = n.classify_quant_repo("mlx-community/DeepSeek-V4-Flash-0731-4bit", "deepseek-ai", "DeepSeek-V4-Flash")
    assert not m.accepted and m.reject_reason.startswith("unknown_suffix:0731")
    m = n.classify_quant_repo("mlx-community/DeepSeek-V4-Flash", "deepseek-ai", "DeepSeek-V4-Flash")
    assert not m.accepted and m.reject_reason == "no_quant_suffix"
    m = n.classify_quant_repo("mlx-community/DeepSeek-V4-Flash-mxfp4", "deepseek-ai", "DeepSeek-V4-Flash")
    assert m.accepted and m.format == "mlx" and m.label == "MLX-MXFP4"


def test_gguf_labels():
    assert n.gguf_label("Qwen3.8-27B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert n.gguf_label("Kimi-K3-UD-IQ3_XXS-00001-of-00012.gguf") == "UD-IQ3_XXS"
    assert n.gguf_label("DeepSeek-V4-Flash-MXFP4_MOE-00001-of-00004.gguf") == "MXFP4_MOE"
    assert n.gguf_label("gemma-4-26b-a4b-it-Q4_0.gguf") == "Q4_0"
    assert n.gguf_label("Qwen3.8-27B-Q4_K_M.gguf") == "Q4_K_M"
    assert n.gguf_label("Qwen3.8-27B-Q6_K.gguf") == "Q6_K"
    assert n.gguf_label("Qwen3.8-27B-Q4_K_L.gguf") == "Q4_K_L"
    assert n.label_bits("UD-Q4_K_XL") == 5.0


def test_quants_from_real_listings(repos):
    uns = repos["unsloth/Qwen3.8-27B-GGUF"]
    total = 27_781_427_952
    quants = n.quants_from_files(uns["siblings"], "gguf", params_total=total)
    labels = {q.label: q for q in quants}
    assert "UD-Q4_K_XL" in labels and "Q8_0" in labels, labels.keys()
    assert all(q.size_bytes > 0 for q in quants)
    assert all(not any("mmproj" in f.lower() for f in q.files) for q in quants)
    assert [q.size_bytes for q in quants] == sorted(q.size_bytes for q in quants)
    ud4 = labels["UD-Q4_K_XL"]
    assert 15.5 <= ud4.size_gb <= 19.0 and 4.3 <= ud4.bits <= 5.5

    bart = repos["bartowski/Qwen3.8-27B-GGUF"]
    bq = n.quants_from_files(bart["siblings"], "gguf", params_total=total)
    q4 = next(q for q in bq if q.label == "Q4_K_M")
    assert 16.0 <= q4.size_gb <= 18.0, q4.size_gb   # ~17-18 GB depending on the quantizer
    assert all("mmproj" not in f.lower() for q in bq for f in q.files)
    assert not any(q.label in ("F16", "BF16") and q.size_gb < 5 for q in bq)   # the projector must not masquerade as a label

    kimi = repos["unsloth/Kimi-K3-GGUF"]
    kq = n.quants_from_files(kimi["siblings"], "gguf")
    assert len(kimi["siblings"]) > 100 and 5 <= len(kq) <= 20          # 171 files collapse into labels
    assert max(q.file_count for q in kq) > 5

    mlx = repos["lmstudio-community/Qwen3.8-27B-MLX-4bit"]
    mq = n.quants_from_files(mlx["siblings"], "mlx", label="MLX-4bit", bits=4.0, params_total=total)
    assert len(mq) == 1 and 15.5 <= mq[0].size_gb <= 16.5                # 16.08 GB

    qat = repos["google/gemma-4-26b-a4b-it-qat-q4_0-gguf"]
    gq = n.quants_from_files(qat["siblings"], "gguf", qat=True)
    assert len(gq) == 1 and gq[0].label == "Q4_0" and gq[0].qat and 13.5 <= gq[0].size_gb <= 15.5


def test_listing_fixture_mostly_parses(listing):
    kinds = [n.strip_suffix(n.split_repo(r["id"])[1]).kind for r in listing]
    assert kinds.count("none") < len(kinds) / 2


def test_sidecar_files_are_skipped():
    files = [
        {"rfilename": "Qwen3.8-27B-Q8_0.gguf", "size": 28_600_000_000},
        {"rfilename": "mmproj-Qwen3.8-27B-Q8_0.gguf", "size": 630_000_000},
        {"rfilename": "mtp-Qwen3.8-27B-Q8_0.gguf", "size": 3_160_000_000},
        {"rfilename": "mtp-Qwen3.8-27B-Q4_0.gguf", "size": 1_680_000_000},
        {"rfilename": "Qwen3.8-27B.imatrix", "size": 10_000_000},
        {"rfilename": "dspark-Qwen3.8-27B-Q8_0.gguf", "size": 2_000_000_000},
        {"rfilename": "dspark/dspark-Qwen3.8-27B-BF16.gguf", "size": 4_000_000_000},
    ]
    qs = n.quants_from_files(files, "gguf")
    assert [(q.label, q.size_bytes) for q in qs] == [("Q8_0", 28_600_000_000)]


def test_duplicate_safetensors_families_count_once():
    files = [{"rfilename": f"consolidated-0000{i}-of-00002.safetensors", "size": 10_000_000_000} for i in (1, 2)]
    files += [{"rfilename": f"model-0000{i}-of-00002.safetensors", "size": 10_000_000_000} for i in (1, 2)]
    q = n.quants_from_files(files, "safetensors", label="FP8", bits=8.0)
    assert len(q) == 1 and q[0].size_bytes == 20_000_000_000 and q[0].file_count == 2


def test_model_names_that_start_with_the_maker_name():
    base_org, stem = "nvidia", "NVIDIA-Nemotron-3-Super-120B-A12B"
    for repo in ("unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF", "bartowski/nvidia_Nemotron-3-Super-120B-A12B-GGUF",
                 "mlx-community/NVIDIA-Nemotron-3-Super-120B-A12B-4bit", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"):
        m = n.classify_quant_repo(repo, base_org, stem)
        assert m.accepted, (repo, m.reject_reason)
    assert n.classify_quant_repo("unsloth/NVIDIA-Nemotron-3-Super-120B-A12B", base_org, stem).reject_reason == "no_quant_suffix"
