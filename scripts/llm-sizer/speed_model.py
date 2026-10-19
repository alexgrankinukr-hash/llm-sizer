"""The fixed-cost ("floor") speed model, mirrored from src/lib/llm-sizer/engine/speed.ts so the fit and the
engine compute the same milliseconds. Every function here has a namesake in speed.ts; the parity fixture
(tests/fixtures/speed-cases.json) pins both sides to the same numbers.

    ms per token = read_ms + fixed_ms + attn_ms
    read_ms      = GB read per token ÷ (b_eff_ratio × spec bandwidth) × 1000 × read_factor[quant family]   (the K-quant factor on Metal only)
    fixed_ms     = (t0_ms + architecture_cost_ms[class]) × mlx_overhead[group]   (GGUF → 1.0)
                   on MLX, a mixture under 6B active instead: expert layers × ms_per_layer[chip generation]
    attn_ms      = (attention_ms_per_32k[attention] + mlx_attention_ms_per_32k[runtime]) × attended ÷ 32768
                   + mlx_index_ms_per_32k[runtime] × context ÷ 32768   (sparse-attention models on MLX only)
    tok/s        = 1000 ÷ ms

GB read per token = active parameters × bits ÷ 8 + cache bytes per token × attended (memory.ts:kvBytesPerToken).
attended          = the context, or a sparse-attention model's budget on MLX (attended_tokens).
"""
from __future__ import annotations

import math
import re

GIB_IN_GB = 1.073741824

# quant families that carry a read factor; everything with K-quant / IQ / UD in its label reads slower on Metal
PLAIN_FAMILIES = {
    "Q4_0": "q4_0", "Q4_1": "q4_0", "Q5_0": "q4_0", "Q5_1": "q4_0",
    "Q8_0": "q8_0", "FP8": "q8_0",
    "F16": "f16", "BF16": "f16", "FP16": "f16", "F32": "f16",
    "MXFP4": "mxfp4", "MXFP4_MOE": "mxfp4", "NVFP4": "mxfp4", "FP4": "mxfp4",
}
ARCH_CLASSES = ("dense", "moe", "moe_hybrid", "moe_latent", "deepseek_v4", "flash_next")
MLX_GROUPS = ("dense", "small_active_moe_m4plus", "small_active_moe_pre_m4", "moe_other")
# the groups whose MLX fixed cost is the expert-layer rule (mlx_expert_ms) when factors carry it
EXPERT_LAYER_GROUPS = ("small_active_moe_m4plus", "small_active_moe_pre_m4")
SMALL_ACTIVE_B = 6.0
CONTEXT_UNIT = 32768
# architectures whose n-gram table Mac engines keep on the SSD when they run an MLX file, with the table's share of the
# file (memory.ts / constants.ts:NGRAM_TABLES): Qwen 3.8 Flash Next, 51B of 180B parameters
NGRAM_TABLES = (("qwen4_exp", 0.283),)
# sparse-attention architectures and the cached tokens their attention reads per generated token on MLX
# (constants.ts:SPARSE_ATTENTION): Qwen 3.8 Flash Next's index picks 2,048 for each attention layer (indexer_budget)
SPARSE_ATTENTION = (("qwen4_exp", 2048),)


def quant_family(label: str, fmt: str | None = None) -> str:
    """mlx | q4_0 | q8_0 | f16 | mxfp4 | gguf_kquant (speed.ts:quantFamilyOf)."""
    if fmt == "mlx" or label.upper().startswith("MLX"):
        return "mlx"
    return PLAIN_FAMILIES.get(label.upper(), "gguf_kquant")


def arch_class(arch: dict | None, override: str | None = None) -> str:
    """dense | moe | moe_hybrid | moe_latent | deepseek_v4 | flash_next (speed.ts:archClassOf)."""
    if override in ARCH_CLASSES:
        return override
    if not arch:
        return "dense"
    # the n-gram-table architecture (Qwen 3.8 Flash Next): 512 experts and a table lookup per token, its own fixed cost
    if str(arch.get("model_type") or "").startswith("qwen4_exp"):
        return "flash_next"
    experts = arch.get("experts_total") or 0
    if experts <= 1:
        return "dense"
    if arch.get("attention") in ("mla", "latent"):
        return "moe_latent"
    if (arch.get("linear_layers") or 0) > 0:
        return "moe_hybrid"
    return "moe"


def ngram_table_share(arch: dict | None) -> float:
    """The share of a file that is the model's n-gram table; 0 for every other architecture (memory.ts:ngramTableShare)."""
    model_type = str((arch or {}).get("model_type") or "")
    return next((share for prefix, share in NGRAM_TABLES if model_type.startswith(prefix)), 0.0)


def table_on_ssd_gb(size_gb: float, fmt: str, arch: dict | None, loader: str = "mlx") -> float:
    """The GB of a plain file's n-gram table that an MLX engine leaves on the SSD; GGUF files and other loaders keep it
    in memory (memory.ts:tableOnSsdGb)."""
    if fmt != "mlx" or loader != "mlx":
        return 0.0
    return size_gb * ngram_table_share(arch)


def sparse_budget(arch: dict | None) -> int | None:
    """The cached tokens a sparse-attention model's attention reads per token, or None (speed.ts:sparseBudgetOf)."""
    model_type = str((arch or {}).get("model_type") or "")
    return next((budget for prefix, budget in SPARSE_ATTENTION if model_type.startswith(prefix)), None)


def attended_tokens(context_tokens: int, runtime: str, budget: int | None) -> int:
    """The context the attention reads per token: the whole of it, or a sparse model's budget under MLX, whose engines
    run the index (speed.ts:attendedTokens). GGUF keeps the whole context."""
    return min(context_tokens, budget) if runtime == "mlx" and budget is not None else context_tokens


def generation_number(chip: str) -> int | None:
    m = re.match(r"^M(\d+)", chip.strip(), re.I)
    return int(m.group(1)) if m else None


def mlx_group(cls: str, active_b: float, chip: str) -> str:
    """Which MLX overhead factor applies (speed.ts:mlxGroupOf)."""
    if cls == "dense":
        return "dense"
    if active_b < SMALL_ACTIVE_B:
        g = generation_number(chip)
        return "small_active_moe_m4plus" if g is not None and g >= 4 else "small_active_moe_pre_m4"
    return "moe_other"


MLX_CONTEXT_GROUPS = ("pre_m5", "m5plus")


def mlx_context_group(chip: str) -> str:
    """Which MLX long-context cost applies (speed.ts:mlxContextGroupOf): the M5 GPU generation pays a different one.
    A chip whose generation cannot be read (a custom machine) is charged the pre-M5 cost, the conservative one."""
    g = generation_number(chip)
    return "m5plus" if g is not None and g >= 5 else "pre_m5"


def gb_per_token(active_b: float, bits: float, kv_bytes_per_token_8bit: int, kv_bits: int, context_tokens: int) -> float:
    return (active_b * 1e9 * bits / 8 + kv_bytes_per_token_8bit * (kv_bits / 8) * context_tokens) / 1e9


def platform_of(chip: str) -> str:
    """The validation rows name the chip only; the DGX Spark is the one CUDA box (speed.ts reads machine.platform)."""
    return "cuda" if chip.strip().lower() == "dgx spark" else "apple"


def read_factor(sm: dict, fam: str, platform: str) -> float:
    """A read factor applies on the platforms it was fitted on (`platforms`); absent means everywhere."""
    t = sm["read_factor"].get(fam)
    if not t:
        return 1.0
    return float(t["value"]) if platform in t.get("platforms", [platform]) else 1.0


def expert_layer_rate(el: dict, chip: str) -> float | None:
    """ms per expert layer for the chip's generation (speed.ts:expertLayerRate): its own fitted value; a generation newer
    than the rows takes the newest fitted one, an older or unreadable one the oldest (the slowest)."""
    rates = el["ms_per_layer"]
    if not rates:
        return None
    g = generation_number(chip)
    key = f"M{g}" if g is not None else None
    if key in rates:
        return rates[key]["value"]
    gens = sorted(int(k[1:]) for k in rates)
    if g is not None and g > gens[-1]:
        return rates[f"M{gens[-1]}"]["value"]
    if g is not None:
        older = [x for x in gens if x < g]
        if older:
            return rates[f"M{older[-1]}"]["value"]
    return rates[f"M{gens[0]}"]["value"]


def mlx_expert_ms(sm: dict, chip: str, moe_layers: int | None) -> float | None:
    """The MLX fixed cost of a small mixture: expert layers × the chip generation's ms per layer (speed.ts:mlxExpertMs).
    None when the factors carry no expert-layer block or the model's layer count is unknown."""
    el = (sm.get("mlx") or {}).get("expert_layer")
    if not el or not moe_layers:
        return None
    rate = expert_layer_rate(el, chip)
    return None if rate is None else moe_layers * rate


def decode_ms(profile: dict, bandwidth_gbs: float, gb: float, fam: str, cls: str, runtime: str, group: str,
              attention: str, context_tokens: int, sm: dict, platform: str = "apple", mlx_ctx: str = "pre_m5",
              chip: str = "", moe_layers: int | None = None, budget: int | None = None) -> float:
    """Milliseconds per generated token under the floor model. `runtime` is 'mlx' or anything else (GGUF path);
    `mlx_ctx` is the chip's MLX context-cost group (mlx_context_group); `chip` and `moe_layers` feed the expert-layer
    rule for small mixtures on MLX; `budget` is a sparse model's (sparse_budget), and `gb` must count the cache over
    attended_tokens(context_tokens, runtime, budget)."""
    b_eff = profile["b_eff_ratio"] * bandwidth_gbs
    read = gb / b_eff * 1000 * read_factor(sm, fam, platform)
    is_mlx = runtime == "mlx"
    expert = mlx_expert_ms(sm, chip, moe_layers) if is_mlx and group in EXPERT_LAYER_GROUPS else None
    if expert is not None:
        fixed = expert
    else:
        fixed = (profile["t0_ms"] + sm["architecture_cost_ms"][cls]["value"]) * (sm["mlx"]["overhead_factor"][group]["value"] if is_mlx else 1.0)
    attn_key = attention if attention in sm["attention_ms_per_32k"] else "gqa"
    mlx_attn = sm["mlx"]["attention_ms_per_32k"]
    mlx_ctx_ms = (mlx_attn.get(mlx_ctx) or mlx_attn["pre_m5"])["value"] if is_mlx else 0.0
    attended = attended_tokens(context_tokens, runtime, budget)
    # a sparse model's index scores every cached token before the attention reads its budget
    index = 0.0
    if is_mlx and budget is not None:
        idx = (sm["mlx"].get("index_ms_per_32k") or {}).get(mlx_ctx)
        index = (idx["value"] if idx else 0.0) * context_tokens / CONTEXT_UNIT
    attn = (sm["attention_ms_per_32k"][attn_key]["value"] + mlx_ctx_ms) * attended / CONTEXT_UNIT + index
    return read + fixed + attn


def legacy_tok_s(eff: float | None, bandwidth_gbs: float, gb: float, runtime: str, is_moe: bool, total_b: float,
                 context_tokens: int, legacy_runtime: dict, fade: tuple[int, int] = (24000, 36000)) -> float | None:
    """The multiplier model as speed.ts ran it before the floor model (speed.ts:estimateSpeed, multiplier branch)."""
    if eff is None or bandwidth_gbs <= 0 or gb <= 0:
        return None
    rf = 1.0
    if runtime == "mlx":
        mlx = legacy_runtime["mlx"]
        base = mlx["moe"]["factor"] if is_moe else (mlx["dense_14b_and_up"]["factor"] if total_b >= 14 else mlx["dense_under_14b"]["factor"])
        lo, hi = fade
        if context_tokens <= lo:
            rf = base
        elif context_tokens >= hi:
            rf = 1.0
        else:
            rf = base + (1 - base) * (context_tokens - lo) / (hi - lo)
    return bandwidth_gbs * eff * rf / gb


def log_err(pred: float | None, meas: float) -> float | None:
    if pred is None or pred <= 0 or meas <= 0:
        return None
    return abs(math.log(pred / meas))


# ---- prefill (prompt processing), mirrored by src/lib/llm-sizer/engine/prefill.ts ---------------------------------

PLAIN_QUANTS = {"Q4_0", "Q8_0", "F16", "BF16", "MXFP4"}


def prefill_constant(pf: dict, chip: str, gpu_cores: int | None, platform: str = "apple") -> tuple[float | None, str, int | None]:
    """K = tokens/s × active billions at a short prompt, and where it comes from: 'measured' (this chip has rows),
    'generation' (its generation's per-core rate on its core count), 'assumed' (the newest generation's rate, for a
    chip newer than the rows), 'none'."""
    if platform == "apple":
        gen = generation_number(chip)
        per_core = pf["per_core"]
        if gen is None or not gpu_cores:
            return None, "none", None
        key = f"M{gen}"
        if key in per_core:
            source = "measured" if chip in pf.get("chips_with_rows", []) else "generation"
            return per_core[key]["value"] * gpu_cores, source, gpu_cores
        newest = max(per_core, key=lambda k: int(k[1:]))
        return per_core[newest]["value"] * gpu_cores, "assumed", gpu_cores
    by_chip = pf["by_chip"]
    if chip in by_chip:
        return by_chip[chip]["value"], "measured", None
    return None, "none", None


def prefill_mlx_group(chip: str) -> str:
    """m1_m2 | m3_m4 | m5plus (prefill.ts:prefillMlxGroupOf); a chip whose generation cannot be read takes m1_m2, the slowest."""
    g = generation_number(chip)
    if g is None:
        return "m1_m2"
    return "m5plus" if g >= 5 else ("m3_m4" if g >= 3 else "m1_m2")


def prefill_tok_s(pf: dict, k: float, quant_label: str, fmt: str, cls: str, runtime: str, active_b: float) -> float:
    """Short-prompt prefill tokens/s for one build on one chip."""
    kq = pf["kquant"]["factor"] if (runtime != "mlx" and fmt != "mlx" and quant_label.upper() not in PLAIN_QUANTS) else 1.0
    class_factor = pf["class"].get(cls, pf["class"]["dense"])["factor"]
    mlx = 1.0
    if runtime == "mlx":
        mlx = pf["mlx"][pf.get("_mlx_group") or "m1_m2"]["factor"]
    return k * kq * class_factor * mlx / active_b


def prefill_d0(pf: dict, kv_layers: int | None, layers: int | None, runtime: str = "gguf", model_id: str | None = None) -> float:
    """The context depth at which the rate has halved (prefill.ts:prefillD0): the runtime's fitted value scaled by the
    model's share of attention layers, or the model's own where it has a published sweep on that runtime."""
    ctx = pf["context"]
    rt = "mlx" if runtime == "mlx" else "gguf"
    own = (ctx.get("by_model") or {}).get(model_id or "", {}).get(rt)
    if own:
        return float(own["d0_tokens"])
    base = ctx["mlx"]["d0_tokens"] if rt == "mlx" and ctx.get("mlx") else ctx["d0_tokens"]
    share = max(0.05, (kv_layers / layers) if (kv_layers and layers) else ctx["attention_share_ref"])
    return base * ctx["attention_share_ref"] / share


def prefill_rate_at(tok_s: float, d0: float, depth: int) -> float:
    return tok_s / (1.0 + depth / d0)


def first_word_wait_s(tok_s: float, d0: float, prompt_tokens: int) -> float:
    """Seconds before the first word for a prompt of n tokens: the marginal rate integrated over the prompt."""
    n = prompt_tokens
    return (n + n * n / (2.0 * d0)) / tok_s


# ---- linked machines, mirrored by src/lib/llm-sizer/engine/cluster.ts ---------------------------------------------

CLUSTER_LINK_CLASSES = ("mac", "spark", "gpu")
CLUSTER_SPLITS = ("layer", "tensor")


def link_class(chip: str, platform: str) -> str:
    """mac | spark | gpu (cluster.ts:linkClassOf). Apple silicon links over Thunderbolt 5; a DGX Spark or an ASUS
    GX10 over its 200 Gb/s ports; anything else is cards on PCIe inside one box."""
    if platform == "apple":
        return "mac"
    c = (chip or "").lower()
    return "spark" if ("spark" in c or "gb10" in c) else "gpu"


def cluster_ms(single_ms: float, n: int, split: str, link: str, kind: str, cf: dict) -> float:
    """Milliseconds per token across n linked machines (cluster.ts:clusterMs). `single_ms` is one machine's time for
    the whole model, `link` a link class, `kind` 'dense' or 'moe'; the constants come from factors.json's cluster block."""
    if n <= 1:
        return single_ms
    if split == "tensor":
        key = "gpu" if link == "gpu" else ("dense" if kind == "dense" else "moe")
        return single_ms / n + cf["tensor_c_ms"][key]["value"] * math.log2(n)
    return single_ms + cf["hop_ms"][link]["value"] * (n - 1)


def prefill_cluster_factor(n: int, split: str, cf: dict) -> float:
    """What linking does to reading speed (cluster.ts:prefillClusterFactor): flat on a layer split, n ^ exponent
    under tensor parallel."""
    if n <= 1:
        return 1.0
    if split == "tensor":
        return float(n) ** cf["prefill"]["tensor_exponent"]["value"]
    return cf["prefill"]["layer_factor"]["value"]
