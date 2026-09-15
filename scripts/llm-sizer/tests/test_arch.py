import arch


def gb(x):
    return round(x / 1e9, 2)


def test_qwen_27b(configs):
    a = arch.classify(configs["qwen3.8-27b"])
    assert (a.attention, a.layers, a.kv_layers, a.linear_layers) == ("gqa", 64, 16, 48)
    assert (a.kv_heads, a.head_dim) == (4, 256)
    assert a.full_bytes_per_token_8bit == 32768
    assert a.accel_mtp and not a.is_moe and a.is_hybrid
    assert 4.2 <= gb(arch.kv_cache_bytes(a, 131072)) <= 4.5
    assert a.native_bits == 16 and a.context_max == 262144 and not a.conventional_assumed


def test_flash_next(configs):
    a = arch.classify(configs["qwen3.8-flash-next"])
    assert (a.kv_layers, a.linear_layers) == (12, 36)
    assert (a.experts_total, a.experts_active, a.experts_shared) == (512, 10, 1)
    assert a.full_bytes_per_token_8bit == 12288 and a.accel_mtp
    active, src = arch.active_params(a, 180_000_000_000)
    assert src == "estimated" and 5e9 < active < 15e9   # the 50 GB n-gram table must not count as active weights


def test_glm_53_flash(configs):
    a = arch.classify(configs["glm-5.3-flash"])
    assert a.attention == "mla" and a.kv_lora_rank == 512 and a.rope_dim is None
    assert a.kv_layers == 11 and a.linear_layers == 34
    assert a.full_bytes_per_token_8bit == 11 * 512
    assert a.accel_mtp and a.native_bits == 8
    assert 0.6 <= gb(arch.kv_cache_bytes(a, 131072)) <= 0.9


def test_glm_52(configs):
    a = arch.classify(configs["glm-5.2"])
    assert a.attention == "mla" and (a.kv_lora_rank, a.rope_dim, a.layers) == (512, 64, 78)
    assert a.full_bytes_per_token_8bit == 78 * 576
    assert 5.7 <= gb(arch.kv_cache_bytes(a, 131072)) <= 6.1


def test_deepseek_v4_flash(configs):
    a = arch.classify(configs["deepseek-v4-flash"])
    assert a.attention == "latent" and a.full_bytes_per_token_8bit == 43 * (512 + 64)
    assert a.accel_mtp and a.native_bits == 4 and a.experts_active == 6
    assert 3.0 <= gb(arch.kv_cache_bytes(a, 131072)) <= 3.4


def test_gemma_26b(configs):
    a = arch.classify(configs["gemma-4-26b-a4b"])
    assert (a.kv_layers, a.sliding_layers, a.window_size) == (5, 25, 1024)
    assert a.k_eq_v and (a.kv_heads, a.head_dim) == (2, 512)
    # the shared K/V projection does not halve the stored cache: K and V are both kept
    assert a.full_bytes_per_token_8bit == 5 * 2 * 1024
    assert a.sliding_bytes_per_token_8bit == 25 * 2 * 8 * 256
    assert a.experts_active == 8 and not a.accel_mtp
    assert 1.3 < gb(arch.kv_cache_bytes(a, 131072)) < 1.6


def test_kimi_k3(configs):
    a = arch.classify(configs["kimi-k3"])
    assert a.attention == "mla" and a.kv_layers == 24 and a.linear_layers == 69
    assert not a.accel_mtp and a.native_bits == 4 and a.experts_active == 16
    assert 1.5 <= gb(arch.kv_cache_bytes(a, 131072)) <= 2.2


def test_unknown_config_is_flagged():
    a = arch.classify({"model_type": "mystery", "num_hidden_layers": 40, "hidden_size": 5120, "num_attention_heads": 40})
    assert a.conventional_assumed and a.source == "config" and a.attention == "gqa"
    assert a.kv_heads == 40 and a.full_bytes_per_token_8bit == 2 * 40 * 128 * 40
    b = arch.classify({"model_type": "mystery"})
    assert b.source == "assumed" and b.attention == "assumed"


def test_moe_with_intermediate_size_only_is_not_dense():
    cfg = {"model_type": "minimax_m3_vl", "num_hidden_layers": 60, "hidden_size": 6144, "num_attention_heads": 48, "num_key_value_heads": 8,
           "head_dim": 128, "intermediate_size": 2048, "num_local_experts": 128, "num_experts_per_tok": 4, "n_shared_experts": 1,
           "moe_layer_freq": [0, 0, 0] + [1] * 57, "max_position_embeddings": 1048576}
    a = arch.classify(cfg)
    assert a.is_moe and a.moe_intermediate == 2048 and a.moe_layers == 57
    active, src = arch.active_params(a, 427_000_000_000)
    assert src == "estimated" and 30e9 < active < 60e9   # intermediate_size accounts for the checkpoint: the exact formula (~40B)


def test_nemotron_h_block_list():
    cfg = {"model_type": "nemotron_h", "hidden_size": 8192, "num_attention_heads": 64, "num_key_value_heads": 2, "head_dim": 128,
           "layers_block_type": ["mamba", "mamba", "moe", "attention"] * 20, "n_routed_experts": 512, "num_experts_per_tok": 22,
           "n_shared_experts": 1, "moe_intermediate_size": 5120, "num_nextn_predict_layers": 1, "max_position_embeddings": 262144, "vocab_size": 131072}
    a = arch.classify(cfg)
    assert a.layers == 80 and a.kv_layers == 20 and a.linear_layers == 40 and a.moe_layers == 20
    assert a.attention == "gqa" and a.full_bytes_per_token_8bit == 20 * 2 * 2 * 128 and a.accel_mtp
    active, src = arch.active_params(a, 560_000_000_000)
    assert src == "estimated" and 30e9 < active < 90e9


def test_minimax_m3_expert_size_from_intermediate(configs):
    a = arch.classify(configs["minimax-m3"])
    assert a.is_moe and a.moe_intermediate == 3072 and a.moe_layers == 57 and a.moe_inter_assumed
    active, src = arch.active_params(a, 427_040_140_160)
    # the card says ~23B; the config-derived count lands near 30B (attention, embeddings and the vision tower are counted in full)
    assert src == "estimated" and 25e9 < active < 35e9


def test_kimi_k3_latent_experts(configs):
    a = arch.classify(configs["kimi-k3"])
    assert a.expert_hidden_size == 3584 and a.hidden_size == 7168
    active, src = arch.active_params(a, 2_779_931_837_184)
    # the card says 104B activated; with the latent width the estimate is ~114B (it was 165B on the hidden size)
    assert src == "estimated" and 100e9 < active < 125e9


def test_qwen3_coder_next(configs):
    a = arch.classify(configs["qwen3-coder-next"])
    assert (a.experts_total, a.experts_active, a.experts_shared, a.linear_layers) == (512, 10, 1, 36)
    active, src = arch.active_params(a, 79_674_391_296)
    assert src == "estimated" and 3.5e9 < active < 4.5e9   # the card's 3B is pinned by an override


def test_gemma_e4b_effective_parameters(configs):
    a = arch.classify(configs["gemma-4-e4b"])
    assert not a.is_moe and a.per_layer_embedding_params == 262144 * 256 * 42
    active, src = arch.active_params(a, 7_996_156_490)
    assert src == "estimated" and 4.4e9 < active < 4.6e9   # 4.5B effective per the card (8B with the embedding tables)
