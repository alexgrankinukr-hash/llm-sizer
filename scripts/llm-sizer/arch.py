"""Read a model's `config.json` into the architecture facts the memory and speed math needs.

Everything is derived from fields; unknown layouts fall back to conventional attention and
are flagged so the UI can say "architecture assumed".
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

# model_type values we have seen and understand; anything else is flagged `conventional_assumed`.
KNOWN_MODEL_TYPES = frozenset({
    "llama", "llama4", "llama4_text", "mistral", "mistral3", "ministral", "mixtral", "phi3", "phi4", "phi4_multimodal",
    "qwen2", "qwen2_moe", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
    "qwen4_exp", "qwen4_exp_text", "gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n", "gemma4", "gemma4_text",
    "gpt_oss", "deepseek", "deepseek_v2", "deepseek_v3", "deepseek_v4", "glm4", "glm4_moe", "glm4v", "glm_moe_dsa",
    "glm5_next", "glm5_next_text", "kimi_k3", "kimi_linear", "kimi_k2", "minimax", "minimax_m1", "minimax_m2", "minimax_m3",
    "nemotron", "nemotron_h", "nemotron_nas", "granite", "granitemoe", "granitemoehybrid", "cohere", "cohere2",
    "olmo2", "olmo3", "smollm3", "falcon_h1", "exaone4", "internlm3", "hunyuan_v1_dense", "hunyuan_v1_moe",
    "seed_oss", "lfm2", "lfm2_moe", "ernie4_5", "ernie4_5_moe", "dots1", "baichuan", "stablelm", "starcoder2",
    "bloom", "gptj", "gpt_neox", "gpt2", "opt", "mimo", "step3", "step3_text", "apertus", "bailing_moe",
})

FULL_TYPES = {"full_attention", "deepseek_sparse_attention", "attention", "global_attention"}
SLIDING_TYPES = {"sliding_attention", "local_attention", "sliding_window"}
LINEAR_TYPES = {"linear_attention", "mamba", "mamba2", "ssm", "recurrent", "kda", "gated_delta_net", "conv"}


def unwrap(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the text-model config (multimodal configs nest it under `text_config`)."""
    inner = cfg.get("text_config")
    if isinstance(inner, dict) and inner:
        merged = dict(inner)
        merged.setdefault("model_type", cfg.get("model_type"))
        merged["_outer_model_type"] = cfg.get("model_type")
        for k in ("quantization_config", "torch_dtype", "dtype"):
            if k in cfg and k not in merged:
                merged[k] = cfg[k]
        return merged
    return cfg


def _first(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        v = cfg.get(k)
        if v is not None:
            return v
    return default


@dataclass
class Arch:
    model_type: str | None
    attention: str                     # gqa | mla | latent | assumed
    layers: int
    kv_layers: int                     # layers with a growing cache
    sliding_layers: int
    linear_layers: int
    kv_heads: int | None
    head_dim: int | None
    k_eq_v: bool
    kv_lora_rank: int | None
    rope_dim: int | None
    window_size: int | None
    full_bytes_per_token_8bit: int     # per token, all growing layers, 8-bit cache
    sliding_bytes_per_token_8bit: int  # per token within the window, all sliding layers
    linear_state_bytes: int            # constant, fp16 state
    hidden_size: int | None
    experts_total: int | None
    experts_active: int | None
    experts_shared: int | None
    moe_intermediate: int | None
    moe_layers: int | None
    accel_mtp: bool
    moe_inter_assumed: bool
    vocab_size: int | None
    context_max: int | None
    native_bits: float | None
    conventional_assumed: bool
    source: str                        # config | assumed
    # the width the routed experts work on when it differs from hidden_size (Kimi K3's latent MoE: routed_expert_hidden_size)
    expert_hidden_size: int | None = None
    # Gemma E-series per-layer embedding tables (vocab_size_per_layer_input × hidden_size_per_layer_input × layers): looked up per token, not streamed
    per_layer_embedding_params: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_moe(self) -> bool:
        return bool(self.experts_total and self.experts_active)

    @property
    def is_hybrid(self) -> bool:
        return self.linear_layers > 0 or self.sliding_layers > 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_moe"] = self.is_moe
        d["is_hybrid"] = self.is_hybrid
        return d


def kv_cache_bytes(arch: Arch, context_tokens: int, kv_bits: int = 8) -> int:
    """Total context-cache bytes at `context_tokens` (grows with context, plus window and linear state)."""
    scale = kv_bits / 8
    grow = arch.full_bytes_per_token_8bit * context_tokens
    window = min(context_tokens, arch.window_size or context_tokens)
    slide = arch.sliding_bytes_per_token_8bit * window
    return int((grow + slide) * scale) + arch.linear_state_bytes


def native_bits(cfg: dict[str, Any]) -> float | None:
    q = cfg.get("quantization_config")
    if isinstance(q, dict):
        method = str(q.get("quant_method", "")).lower()
        fmt = str(q.get("format", "")).lower()
        if "mxfp4" in method or "mxfp4" in fmt:
            return 4.0
        if "fp8" in method or "fp8" in fmt:
            expert = str(cfg.get("expert_dtype", "")).lower()
            return 4.0 if "fp4" in expert else 8.0
        if "nvfp4" in method or "fp4" in method or "fp4" in fmt:
            return 4.0
        if "compressed-tensors" in method:
            groups = json.dumps(q.get("config_groups", {})).lower()
            if "mxfp4" in groups or "fp4" in groups:
                return 4.0
            if "fp8" in groups or "float8" in groups:
                return 8.0
            if "int4" in groups or "w4" in groups:
                return 4.0
            if "int8" in groups or "w8" in groups:
                return 8.0
        if "awq" in method or "gptq" in method:
            return float(q.get("bits") or 4)
    dtype = str(_first(cfg, "torch_dtype", "dtype", default="")).lower()
    if "float32" in dtype:
        return 32.0
    if "bfloat16" in dtype or "float16" in dtype:
        return 16.0
    return None


def mtp_flag(cfg: dict[str, Any]) -> bool:
    if (cfg.get("num_nextn_predict_layers") or 0) >= 1:
        return True
    if (cfg.get("mtp_num_hidden_layers") or 0) >= 1:
        return True
    return bool(cfg.get("mtp"))


def classify(raw_cfg: dict[str, Any]) -> Arch:
    cfg = unwrap(raw_cfg)
    notes: list[str] = []
    model_type = cfg.get("model_type")
    outer = cfg.get("_outer_model_type")
    known = (model_type in KNOWN_MODEL_TYPES) or (outer in KNOWN_MODEL_TYPES)

    block_types = cfg.get("layers_block_type") if isinstance(cfg.get("layers_block_type"), list) else None
    pattern = cfg.get("hybrid_override_pattern") if isinstance(cfg.get("hybrid_override_pattern"), str) else None
    layers = _first(cfg, "num_hidden_layers", "n_layer", "num_layers")
    if not layers and block_types:
        layers = len(block_types)
    elif not layers and pattern:
        layers = len(pattern)
    hidden = _first(cfg, "hidden_size", "n_embd", "d_model")
    n_heads = _first(cfg, "num_attention_heads", "n_head")
    kv_heads = _first(cfg, "num_key_value_heads")
    head_dim = _first(cfg, "head_dim")
    assumed = False
    if not layers or not hidden or not n_heads:
        assumed = True
        layers = layers or 32
        hidden = hidden or 4096
        n_heads = n_heads or 32
        notes.append("core fields missing; conventional defaults used")
    if kv_heads is None:
        kv_heads = n_heads
        if not assumed:
            notes.append("num_key_value_heads missing; assumed multi-head (no GQA)")
    if not head_dim:
        head_dim = hidden // n_heads

    # --- layer layout ---------------------------------------------------------------
    full = sliding = linear = 0
    layer_types = cfg.get("layer_types")
    lin_cfg = cfg.get("linear_attn_config") if isinstance(cfg.get("linear_attn_config"), dict) else None
    if block_types or pattern:
        # Nemotron-H style: every block is one of attention / mamba / mlp / moe; only attention blocks keep a cache
        blocks = block_types or [{"*": "attention", "M": "mamba", "-": "mlp", "E": "moe"}.get(ch, ch) for ch in pattern]
        for t in blocks:
            t = str(t).lower()
            if t in ("attention", "attn", "*"):
                full += 1
            elif t in LINEAR_TYPES or t.startswith("mamba"):
                linear += 1
        layers = len(blocks)
        notes.append("block list: only attention blocks counted for the cache")
    elif isinstance(layer_types, list) and layer_types:
        for t in layer_types:
            t = str(t)
            if t in FULL_TYPES:
                full += 1
            elif t in SLIDING_TYPES:
                sliding += 1
            elif t in LINEAR_TYPES:
                linear += 1
            else:
                full += 1
                notes.append(f"unknown layer type '{t}' counted as full attention")
        if len(layer_types) != layers:
            notes.append(f"layer_types has {len(layer_types)} entries for {layers} layers")
    elif lin_cfg and isinstance(lin_cfg.get("full_attn_layers"), list):
        full = len(lin_cfg["full_attn_layers"])
        linear = layers - full
    elif cfg.get("full_attention_interval"):
        full = layers // int(cfg["full_attention_interval"])
        linear = layers - full
    elif cfg.get("sliding_window") and cfg.get("sliding_window_pattern"):
        pattern = int(cfg["sliding_window_pattern"])
        full = layers // pattern
        sliding = layers - full
    else:
        full = layers

    # --- attention scheme per growing layer -------------------------------------------
    kv_lora = cfg.get("kv_lora_rank")
    rope = cfg.get("qk_rope_head_dim")
    k_eq_v = bool(cfg.get("attention_k_eq_v"))
    window = cfg.get("sliding_window") if sliding else None
    if kv_lora:
        attention = "mla"
        per_layer = int(kv_lora) + int(rope or 0)
        eff_kv_heads, eff_head_dim = 1, per_layer
    elif kv_heads == 1 and head_dim >= 512:
        attention = "latent"
        per_layer = int(head_dim) + int(rope or 0)
        eff_kv_heads, eff_head_dim = 1, per_layer
    else:
        attention = "assumed" if assumed else "gqa"
        g_heads = cfg.get("num_global_key_value_heads")
        g_dim = cfg.get("global_head_dim")
        if g_heads and g_dim:
            eff_kv_heads, eff_head_dim = int(g_heads), int(g_dim)
            notes.append("full-attention layers use global KV heads")
        else:
            eff_kv_heads, eff_head_dim = int(kv_heads), int(head_dim)
        # attention_k_eq_v (Gemma 4) shares one projection for keys and values; the runtime still stores both,
        # so the cache is never halved. k_eq_v stays on the record as an informational flag.
        per_layer = 2 * eff_kv_heads * eff_head_dim
    full_bytes = per_layer * full
    sliding_bytes = 2 * int(kv_heads) * int(head_dim) * sliding

    # --- linear / recurrent state (constant) --------------------------------------
    linear_state = 0
    if linear:
        v_heads = _first(cfg, "linear_num_value_heads")
        k_dim = _first(cfg, "linear_key_head_dim")
        v_dim = _first(cfg, "linear_value_head_dim")
        if v_heads and k_dim and v_dim:
            linear_state = int(v_heads) * int(k_dim) * int(v_dim) * 2 * linear
        elif lin_cfg and lin_cfg.get("num_heads") and lin_cfg.get("head_dim"):
            hd = int(lin_cfg["head_dim"])
            linear_state = int(lin_cfg["num_heads"]) * hd * hd * 2 * linear
        else:
            linear_state = int(hidden) * int(hidden) // 8 * linear  # rough placeholder
            notes.append("linear-layer state size estimated")

    # --- MoE ----------------------------------------------------------------------
    experts_total = _first(cfg, "num_experts", "n_routed_experts", "num_local_experts")
    experts_active = _first(cfg, "num_experts_per_tok", "num_experts_per_token", "top_k_experts", "moe_top_k")
    shared = _first(cfg, "n_shared_experts", "num_shared_experts")
    if shared is None:
        shared = 1 if cfg.get("shared_expert_intermediate_size") else 0
    moe_inter = _first(cfg, "moe_intermediate_size", "expert_intermediate_size")
    moe_layers = None
    moe_inter_assumed = False
    if experts_total and experts_active:
        if not moe_inter:
            moe_inter = _first(cfg, "intermediate_size")   # MiniMax, Mixtral: one intermediate size, used per expert
            moe_inter_assumed = True
            notes.append("expert size taken from intermediate_size")
        dense_first = int(_first(cfg, "first_k_dense_replace", default=0) or 0)
        moe_layers = int(layers) - dense_first
        mlp_types = cfg.get("mlp_layer_types")
        freq = cfg.get("moe_layer_freq")
        if isinstance(mlp_types, list) and mlp_types:
            moe_layers = sum(1 for t in mlp_types if "moe" in str(t) or "sparse" in str(t))
        elif isinstance(freq, list) and freq:
            moe_layers = sum(1 for t in freq if t)
        elif block_types:
            moe_layers = sum(1 for t in block_types if str(t).lower() in ("moe", "e"))
    else:
        experts_total = experts_active = None
        moe_inter = None
        shared = None
    expert_hidden = _first(cfg, "routed_expert_hidden_size", "moe_hidden_size") if experts_total else None
    if expert_hidden and int(expert_hidden) != int(hidden):
        notes.append(f"experts work on a {int(expert_hidden)}-wide latent, not the {int(hidden)} hidden size")
    ple_vocab = _first(cfg, "vocab_size_per_layer_input")
    ple_hidden = _first(cfg, "hidden_size_per_layer_input")
    per_layer_embedding = int(ple_vocab) * int(ple_hidden) * int(layers) if ple_vocab and ple_hidden else None
    if per_layer_embedding:
        notes.append("per-layer embedding tables are looked up, not streamed; effective parameters exclude them")

    return Arch(
        model_type=model_type, attention=attention, layers=int(layers), kv_layers=full, sliding_layers=sliding,
        linear_layers=linear, kv_heads=eff_kv_heads if attention == "gqa" else int(kv_heads), head_dim=eff_head_dim if attention == "gqa" else int(head_dim),
        k_eq_v=k_eq_v, kv_lora_rank=int(kv_lora) if kv_lora else None, rope_dim=int(rope) if rope else None,
        window_size=int(window) if window else None, full_bytes_per_token_8bit=int(full_bytes),
        sliding_bytes_per_token_8bit=int(sliding_bytes), linear_state_bytes=int(linear_state), hidden_size=int(hidden),
        experts_total=int(experts_total) if experts_total else None, experts_active=int(experts_active) if experts_active else None,
        experts_shared=int(shared) if shared is not None else None, moe_intermediate=int(moe_inter) if moe_inter else None,
        moe_layers=moe_layers, accel_mtp=mtp_flag(cfg), moe_inter_assumed=moe_inter_assumed, vocab_size=_first(cfg, "vocab_size"),
        context_max=_first(cfg, "max_position_embeddings", "n_positions"),
        native_bits=native_bits(cfg), conventional_assumed=(not known) or assumed, source="assumed" if assumed else "config",
        expert_hidden_size=int(expert_hidden) if expert_hidden else None, per_layer_embedding_params=per_layer_embedding,
        notes=notes,
    )


def active_params(arch: Arch, total_params: int) -> tuple[int, str]:
    """Estimate parameters touched per token, the way makers count them. Returns (count, source) where source is
    'dense' or 'estimated'. A published figure in models.overrides.json always wins over this estimate."""
    if not arch.is_moe or not arch.moe_intermediate or not arch.moe_layers:
        if arch.per_layer_embedding_params:
            # Gemma E-series: the per-layer embedding tables and the token embedding are lookups; the makers' "effective"
            # count (E4B: 4.5B of 8B) leaves both out, and that is what is re-read per token
            emb = int(arch.vocab_size or 0) * int(arch.hidden_size or 0)
            return max(int(total_params - arch.per_layer_embedding_params - emb), int(0.3 * total_params)), "estimated"
        return total_params, "dense"
    shared = arch.experts_shared or 0
    width = arch.expert_hidden_size or arch.hidden_size
    per_expert = 3 * width * arch.moe_intermediate
    expert_total = arch.experts_total * per_expert * arch.moe_layers
    if arch.moe_inter_assumed and not (0.5 * total_params <= expert_total <= 1.05 * total_params):
        # expert size unknown and intermediate_size does not account for the checkpoint: assume ~8 % of the weights are
        # shared (attention, embeddings) and the rest is experts
        frac = (arch.experts_active + shared) / (arch.experts_total + shared)
        return int(total_params * (0.08 + 0.92 * frac)), "estimated"
    non_expert = total_params - expert_total
    # A transformer's non-expert weights cannot exceed roughly 12·h² per layer plus the embeddings; anything above that
    # is something else in the checkpoint (an n-gram table, vision tower) that is not read for every token.
    cap = 12 * arch.hidden_size * arch.hidden_size * arch.layers + 2 * (arch.vocab_size or 0) * arch.hidden_size
    non_expert = max(min(non_expert, cap), int(0.02 * total_params))
    active_expert = (arch.experts_active + shared) * per_expert * arch.moe_layers
    return int(non_expert + active_expert), "estimated"


def stripped_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """The unwrapped config minus bulky/irrelevant keys, for storage."""
    cfg = dict(unwrap(raw_cfg))
    for k in ("quantization_config", "_outer_model_type", "id2label", "label2id", "auto_map", "architectures"):
        cfg.pop(k, None)
    return cfg


def config_sha(raw_cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(raw_cfg, sort_keys=True).encode()).hexdigest()[:16]
