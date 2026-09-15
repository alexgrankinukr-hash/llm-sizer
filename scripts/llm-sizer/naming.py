"""Repository-name rules for Hugging Face model and quantization repos.

The quantizers people actually use publish one repo per (model, format): `<stem>-GGUF`,
`<stem>-MLX-4bit`, `<stem>-4bit`, and the makers publish precision variants like
`<stem>-FP8`. Everything here is pure string logic so it can be unit-tested offline.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

QUANTIZER_ORGS: tuple[str, ...] = ("unsloth", "bartowski", "lmstudio-community", "mlx-community", "ggml-org")

# Tokens that mark a modified variant we never treat as "the model" (matched on `-`/`_` tokens, not substrings).
REJECT_TOKENS = frozenset({
    "uncensored", "abliterated", "heretic", "derisked", "reap", "optiq", "dq", "mixed",
    "mtp", "dflash", "dspark", "draft", "speculator", "eagle", "unquantized", "layers", "pruned", "distill", "distilled",
})
_QUANT_LABEL_TOKEN = re.compile(r"^(ud|iq\d.*|q\d(_.*)?|w\d+a\d+|gptq|awq|exl\d|bnb)$", re.I)


def looks_like_quant_name(name: str) -> bool:
    """A repo whose *name* carries a quant label (`…-UD-IQ3_S-layers`) is never a base model."""
    return any(_QUANT_LABEL_TOKEN.match(t) for t in tokens(name))
INSTRUCT_TOKENS = frozenset({"it", "instruct", "chat"})
BASE_TOKENS = frozenset({"base", "pt"})
THINKING_TOKENS = frozenset({"thinking", "reasoning", "think"})

PRECISION_BITS = {"fp8": 8.0, "mxfp8": 8.0, "nvfp4": 4.0, "mxfp4": 4.0, "fp4": 4.0,
                  "bf16": 16.0, "fp16": 16.0, "f16": 16.0, "fp32": 32.0, "f32": 32.0}

# Approximate bits per weight for GGUF labels (llama.cpp); exact values come from file size ÷ params when known.
LABEL_BITS = {
    "IQ1_S": 1.56, "IQ1_M": 1.75, "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "IQ2_M": 2.7,
    "Q2_K": 2.63, "Q2_K_S": 2.5, "Q2_K_L": 3.0, "IQ3_XXS": 3.06, "IQ3_XS": 3.3, "IQ3_S": 3.44, "IQ3_M": 3.66,
    "Q3_K_S": 3.44, "Q3_K_M": 3.91, "Q3_K_L": 4.27, "Q3_K_XL": 4.5, "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "Q4_0": 4.5, "Q4_1": 5.0, "Q4_K_S": 4.58, "Q4_K_M": 4.85, "Q4_K_L": 5.0, "Q4_K_XL": 5.0,
    "Q5_0": 5.5, "Q5_1": 6.0, "Q5_K_S": 5.5, "Q5_K_M": 5.69, "Q5_K_L": 5.9, "Q5_K_XL": 5.9,
    "Q6_K": 6.56, "Q6_K_L": 6.7, "Q6_K_XL": 6.7, "Q8_0": 8.5, "Q8_K_XL": 8.6,
    "MXFP4": 4.25, "MXFP4_MOE": 4.25, "FP8": 8.0, "BF16": 16.0, "F16": 16.0, "F32": 32.0,
}

GGUF_LABEL = re.compile(r"(?i)(?<![A-Z0-9])(UD-)?(IQ\d_[A-Z0-9_]+?|Q\d_K_XL|Q\d_K_L|Q\d_K_[SM]|Q\d_K|Q\d_\d|MXFP4(?:_MOE)?|FP8|BF16|F16|F32)(?=[.\-_]|$)")
SHARD = re.compile(r"-\d{5}-of-\d{5}")
_SEP = re.compile(r"[-_]+")


@dataclass
class Suffix:
    kind: str                      # gguf | mlx | precision | none
    stem: str
    bits: float | None = None
    label: str | None = None
    qat: bool = False


_SUFFIX_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("qat_gguf", re.compile(r"^(?P<stem>.+?)-qat-(?P<q>q\d_\d|int4)-gguf$", re.I)),
    ("gguf", re.compile(r"^(?P<stem>.+?)-gguf$", re.I)),
    ("mlx_bits", re.compile(r"^(?P<stem>.+?)-mlx-(?P<bits>\d+(?:\.\d+)?)bit$", re.I)),
    ("mlx_bits", re.compile(r"^(?P<stem>.+?)-(?P<bits>\d+(?:\.\d+)?)bit$", re.I)),
    ("mlx_prec", re.compile(r"^(?P<stem>.+?)-mlx-(?P<prec>bf16|fp16|fp8|fp32)$", re.I)),
    ("precision", re.compile(r"^(?P<stem>.+?)-(?P<prec>fp8|mxfp8|nvfp4|mxfp4|fp4|bf16|fp16|fp32)$", re.I)),
]


def strip_suffix(name: str) -> Suffix:
    """Split a repo name into (stem, format hint). `name` is the part after the org."""
    for kind, rx in _SUFFIX_RULES:
        m = rx.match(name)
        if not m:
            continue
        stem = m.group("stem")
        if kind == "qat_gguf":
            return Suffix("gguf", stem, None, None, qat=True)
        if kind == "gguf":
            return Suffix("gguf", stem)
        if kind == "mlx_bits":
            bits = float(m.group("bits"))
            label = f"MLX-{m.group('bits')}bit"
            return Suffix("mlx", stem, bits, label)
        if kind == "mlx_prec":
            prec = m.group("prec").lower()
            return Suffix("mlx", stem, PRECISION_BITS[prec], f"MLX-{prec.upper()}")
        if kind == "precision":
            prec = m.group("prec").lower()
            return Suffix("precision", stem, PRECISION_BITS[prec], prec.upper())
    return Suffix("none", name)


def tokens(name: str) -> list[str]:
    return [t for t in _SEP.split(name.lower()) if t]


def reject_token(name: str) -> str | None:
    for t in tokens(name):
        if t in REJECT_TOKENS:
            return t
    return None


def stem_key(stem: str) -> str:
    """Canonical comparison key: lowercase, `-`/`_` unified, instruct/chat tokens removed."""
    toks = [t for t in tokens(stem) if t not in INSTRUCT_TOKENS]
    return "-".join(toks)


def same_model(a: str, b: str) -> bool:
    return stem_key(a) == stem_key(b)


def variant(stem: str) -> str:
    toks = set(tokens(stem))
    if toks & THINKING_TOKENS:
        return "thinking"
    if toks & BASE_TOKENS:
        return "base"
    return "instruct"


def slugify(stem: str) -> str:
    s = re.sub(r"[^a-z0-9.]+", "-", stem.lower()).strip("-.")
    s = re.sub(r"-{2,}", "-", s)
    return s


def split_repo(repo_id: str) -> tuple[str, str]:
    org, _, name = repo_id.partition("/")
    return org, name


@dataclass
class Match:
    repo_id: str
    accepted: bool
    format: str | None = None           # gguf | mlx | safetensors
    label: str | None = None            # MLX-4bit / FP8; GGUF labels come from the files
    bits: float | None = None
    qat: bool = False
    quantizer: str | None = None        # org name, or 'official' for the maker's own repo
    variant: str = "instruct"
    reject_reason: str | None = None
    stem: str | None = None


def classify_quant_repo(repo_id: str, base_org: str, base_stem: str) -> Match:
    """Decide whether `repo_id` is a quantization of `<base_org>/<base_stem>` and, if so, how."""
    org, name = split_repo(repo_id)
    bad = reject_token(name)
    if bad:
        return Match(repo_id, False, reject_reason=f"variant_token:{bad}")

    # Org prefixes: bartowski writes `<org>_<Name>`, mlx-community and others `<org>-<Name>`.
    # A model name can itself start with the maker's name (`NVIDIA-Nemotron-…`), so try both readings.
    candidates = [name]
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9.\-]*)_(.+)$", name)
    if org == "bartowski" and m:
        prefix, rest = m.group(1), m.group(2)
        if prefix.lower() != base_org.lower():
            return Match(repo_id, False, reject_reason=f"foreign_prefix:{prefix}")
        candidates = [rest]
    elif org != base_org and name.lower().startswith(base_org.lower() + "-"):
        candidates = [name[len(base_org) + 1:], name]

    # Quantizers sometimes drop a maker prefix that is part of the model name (`bartowski/nvidia_Nemotron-…`
    # for `NVIDIA-Nemotron-…`), so the base stem is accepted with or without that leading org token.
    base_keys = {stem_key(base_stem)}
    if stem_key(base_stem).startswith(base_org.lower() + "-"):
        base_keys.add(stem_key(base_stem)[len(base_org) + 1:])

    suf = strip_suffix(candidates[0])
    for cand in candidates:
        s2 = strip_suffix(cand)
        if s2.kind != "none" and stem_key(s2.stem) in base_keys:
            suf = s2
            break
    if suf.kind == "none":
        return Match(repo_id, False, reject_reason="no_quant_suffix", stem=suf.stem)
    if stem_key(suf.stem) not in base_keys:
        key, base_key = stem_key(suf.stem), stem_key(base_stem)
        if key.endswith("-" + base_key):
            return Match(repo_id, False, reject_reason="foreign_prefix:" + key[: -len(base_key) - 1], stem=suf.stem)
        if key.startswith(base_key + "-"):
            return Match(repo_id, False, reject_reason="unknown_suffix:" + key[len(base_key) + 1:], stem=suf.stem)
        return Match(repo_id, False, reject_reason="stem_mismatch", stem=suf.stem)

    fmt = {"gguf": "gguf", "mlx": "mlx", "precision": "safetensors"}[suf.kind]
    if suf.kind == "precision" and org == "mlx-community":
        fmt, label = "mlx", f"MLX-{suf.label}"
    else:
        label = suf.label
    quantizer = "official" if org.lower() == base_org.lower() else org
    return Match(repo_id, True, fmt, label, suf.bits, suf.qat, quantizer, variant(suf.stem), stem=suf.stem)


def gguf_label(filename: str) -> str | None:
    base = filename.rsplit("/", 1)[-1]
    m = GGUF_LABEL.search(base)
    if not m:
        return None
    return (m.group(1) or "").upper() + m.group(2).upper()


def label_bits(label: str) -> float | None:
    core = label[3:] if label.startswith("UD-") else label
    return LABEL_BITS.get(core)


@dataclass
class Quant:
    format: str
    label: str
    bits: float | None
    size_bytes: int
    file_count: int
    qat: bool = False
    files: list[str] = field(default_factory=list)

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / 1e9, 2)


SIDECAR_PREFIXES = ("mmproj", "mtp-", "draft-", "eagle", "imatrix", "tokenizer", "vocab", "dspark", "dflash")
SIDECAR_TOKENS = ("mmproj", "imatrix", "-mtp-", "-mtp.", "-draft-", "projector", "vision-encoder", "-dflash", "-dspark")


def _is_sidecar(basename: str) -> bool:
    """Files that live in a quant repo but are not the model weights (vision projectors, MTP heads, drafts)."""
    return basename.startswith(SIDECAR_PREFIXES) or any(t in basename for t in SIDECAR_TOKENS)


def quants_from_files(files: list[dict], fmt: str, label: str | None = None, bits: float | None = None,
                      params_total: int | None = None, qat: bool = False) -> list[Quant]:
    """Turn a repo file listing (`[{rfilename, size}]`) into quant records.

    GGUF repos hold many labels (shards summed per label, `mmproj*`/imatrix skipped);
    MLX and precision repos hold one set of safetensors = one record.
    """
    out: list[Quant] = []
    if fmt == "gguf":
        sizes: dict[str, int] = defaultdict(int)
        counts: dict[str, int] = defaultdict(int)
        names: dict[str, list[str]] = defaultdict(list)
        for f in files:
            name = f["rfilename"]
            low = name.lower()
            base = low.rsplit("/", 1)[-1]
            if not low.endswith(".gguf") or _is_sidecar(base):
                continue
            lab = gguf_label(SHARD.sub("", name))
            if not lab:
                continue
            sizes[lab] += int(f.get("size") or 0)
            counts[lab] += 1
            names[lab].append(name)
        for lab, size in sizes.items():
            b = round(size * 8 / params_total, 2) if params_total else label_bits(lab)
            out.append(Quant("gguf", lab, b, size, counts[lab], qat, names[lab]))
    else:
        weight_files = [f for f in files if f["rfilename"].lower().endswith((".safetensors", ".npz")) and not _is_sidecar(f["rfilename"].lower().rsplit("/", 1)[-1])]
        # Some makers ship the same weights twice (Mistral: `consolidated-*.safetensors` + HF-format `model-*.safetensors`).
        if any(f["rfilename"].lower().rsplit("/", 1)[-1].startswith("consolidated") for f in weight_files) and \
           any(not f["rfilename"].lower().rsplit("/", 1)[-1].startswith("consolidated") for f in weight_files):
            weight_files = [f for f in weight_files if not f["rfilename"].lower().rsplit("/", 1)[-1].startswith("consolidated")]
        size = sum(int(f.get("size") or 0) for f in weight_files)
        if size > 0 and label:
            b = round(size * 8 / params_total, 2) if params_total else bits
            out.append(Quant(fmt, label, b, size, len(weight_files), qat, [f["rfilename"] for f in weight_files]))
    out.sort(key=lambda q: q.size_bytes)
    return out


def strip_precision_variant(repo_name: str) -> str:
    """`Qwen3.8-27B-FP8` -> `Qwen3.8-27B` (used when a listing's top repo is the FP8 mirror)."""
    suf = strip_suffix(repo_name)
    return suf.stem if suf.kind == "precision" else repo_name
