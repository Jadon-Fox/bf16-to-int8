"""Offline model → INT8 pin.
Default hop: dense BF16/F16/F32 → INT8 (schema bf16_to_int8_pin_v1).
NF4/FP4 sources are a lossy second hop (HARD_BLOCK unless --allow-requant).
Dest ABI: bf16_to_int8_pin_v1 (legacy string nf4_to_int8_pin_v1 still loads).
One-time. Not dest-pack. Not orch hot path.
"""
from __future__ import annotations
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
_GPU_QUANT_LOGGED = False
from dtype_io import numel_of, unpack_dense
from nf4 import (
    NF4_CODEBOOK,
    NIBBLE_LO_THEN_HI,
    decode_absmax,
    dequant_int8,
    dequant_nf4,
    max_abs_err,
    pack_f32,
    quantize_int8_symmetric,
    quantize_nf4,
    qweight_nbytes,
    rmse,
    unpack_f32,
    pack_indices,
)
from nested_nf import NF8_CELLS, cells_flat, decode_nested, encode_nested
from safetensors_io import SafeTensorsFile, SafeTensorsSet, write_safetensors
COPY_SIDE_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
)
NF4_SUFFIXES = (
    ".weight.absmax",
    ".weight.nested_absmax",
    ".weight.quant_map",
    ".weight.nested_quant_map",
    ".weight.quant_state.bitsandbytes__nf4",
    ".weight.quant_state.bitsandbytes__fp4",
    ".weight.quant_state",
)
REFUSE_SUFFIXES = (".qweight", ".qzeros", ".g_idx", ".qzeros_zeros")
REFUSE_METHODS = {"gptq", "awq", "gguf", "squeezellm", "hqq", "spqr", "aqlm", "eetq"}
KEEP_SUBSTR = (
    "bias",
    "layernorm",
    "layer_norm",
    "rms_norm",
    "rmsnorm",
    "ln_f",
    "input_layernorm",
    "post_attention_layernorm",
    "norm.weight",
    "norm.bias",
    "inv_freq",
    "rotary",
    "cos_cached",
    "sin_cached",
)
EMBED_SUBSTR = (
    "embed_tokens",
    "wte",
    "wpe",
    "tok_embeddings",
    "word_embeddings",
    "token_embd",
    "embed_in",
    "model.embed",
)
DENSE_DTYPES = ("BF16", "F16", "F32")
@dataclass
class Policy:
    dest: str = "int8" # int8 | nf4 | nested-nf8
    allow_requant: bool = False # NF4/GPTQ already quantized → refuse unless set
    dense: str = "quantize" # quantize | copy (16-bit linears)
    embed: str = "copy" # copy | quantize
    # norms always copy
INT8_PIN_SCHEMA = "bf16_to_int8_pin_v1"
INT8_PIN_SCHEMA_LEGACY = "nf4_to_int8_pin_v1"
INT8_PIN_QUANT_METHOD = "bf16_to_int8_pin"
def _dest_schema(dest: str) -> str:
    if dest == "nf4":
        return "bf16_to_nf4_pin_v1"
    if dest == "nested-nf8":
        return "nested_nf8_pin_v1"
    return INT8_PIN_SCHEMA # orch INT8 loader ABI (legacy nf4_to_int8_pin_v1 still loads)
def find_weight_files(src: Path) -> List[Path]:
    """Single file, or HF snapshot shards via model.safetensors.index.json."""
    if src.is_file() and src.suffix == ".safetensors":
        return [src]
    if src.is_file() and src.suffix == ".gguf":
        raise ValueError("GGUF is not v1 — convert to safetensors first")
    if not src.is_dir():
        raise FileNotFoundError(f"src not found: {src}")
    gguf = list(src.glob("*.gguf"))
    st_files = list(src.glob("*.safetensors"))
    if gguf and not st_files:
        raise ValueError("GGUF is not v1 — convert to safetensors first")
    idx = src / "model.safetensors.index.json"
    if idx.is_file():
        data = json.loads(idx.read_text())
        names = sorted(set((data.get("weight_map") or {}).values()))
        if not names:
            raise ValueError("empty weight_map in model.safetensors.index.json")
        paths = []
        for n in names:
            p = src / n
            if not p.is_file():
                raise FileNotFoundError(f"shard missing: {p}")
            paths.append(p)
        return paths
    for name in ("model.safetensors", "model-00001-of-00001.safetensors"):
        p = src / name
        if p.is_file():
            return [p]
    shards = sorted(src.glob("*.safetensors"))
    if len(shards) == 1:
        return shards
    if len(shards) > 1:
        return shards
    raise FileNotFoundError(f"no .safetensors in {src}")
def find_weight_file(src: Path) -> Path:
    """Back-compat: first weight file (single-file callers)."""
    return find_weight_files(src)[0]
def _stem_of_nf4_aux(name: str) -> Optional[str]:
    for suf in NF4_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return None
def group_modules(names: List[str]) -> Tuple[List[str], List[str], List[str]]:
    aux = {}
    for n in names:
        stem = _stem_of_nf4_aux(n)
        if stem is not None:
            aux.setdefault(stem, []).append(n)
    nf4 = []
    consumed = []
    for stem, extras in aux.items():
        w = stem + ".weight"
        if w in names:
            nf4.append(stem)
            consumed.extend(extras)
            consumed.append(w)
    consumed_set = set(consumed)
    passthrough = [n for n in names if n not in consumed_set]
    return sorted(nf4), passthrough, consumed
def _parse_quant_state(raw: bytes) -> dict:
    text = raw.decode("utf-8", errors="strict").strip().rstrip("\x00")
    return json.loads(text)
def _is_keep(name: str) -> bool:
    n = name.lower()
    return any(s in n for s in KEEP_SUBSTR)
def _is_embed(name: str) -> bool:
    n = name.lower()
    if n.endswith("lm_head.weight") or n.endswith("output.weight"):
        return True
    return any(s in n for s in EMBED_SUBSTR)
def _is_linear_weight(name: str, shape: Tuple[int, ...], dtype: str) -> bool:
    if dtype not in DENSE_DTYPES:
        return False
    if not name.endswith(".weight"):
        return False
    if len(shape) != 2:
        return False
    if _is_keep(name) or _is_embed(name):
        return False
    return True
def refuse_reasons(st: SafeTensorsFile, src_dir: Path) -> List[str]:
    hits = []
    for n in st.names():
        ln = n.lower()
        if any(ln.endswith(s) or s[1:] in ln for s in REFUSE_SUFFIXES if s.startswith(".")):
            if any(x in ln for x in ("qweight", "qzeros", "g_idx")):
                hits.append(f"GPTQ/AWQ-like tensor {n}")
                break
    cfg_path = src_dir / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            cfg = {}
        q = cfg.get("quantization_config") or {}
        method = str(q.get("quant_method") or q.get("quant_type") or "").lower()
        if method in REFUSE_METHODS:
            hits.append(f"config.json quant_method={method} not v1")
    return hits
def convert_nf4_module(
    st: SafeTensorsFile,
    stem: str,
    int8_blocksize: int,
    nibble_order: int = NIBBLE_LO_THEN_HI,
) -> Tuple[bytes, List[float], dict]:
    w_name = stem + ".weight"
    info = st.tensors[w_name]
    qweight = st.read_bytes(w_name)
    state: dict = {}
    for tag in (
        stem + ".weight.quant_state.bitsandbytes__nf4",
        stem + ".weight.quant_state.bitsandbytes__fp4",
    ):
        if tag in st.tensors:
            state = _parse_quant_state(st.read_bytes(tag))
            break
    shape = tuple(state.get("shape") or [])
    if len(shape) != 2:
        packed = 1
        for d in info.shape:
            packed *= int(d)
        n_elem = packed * 2
        shape = (n_elem, 1)
    else:
        n_elem = int(shape[0]) * int(shape[1])
    blocksize = int(state.get("blocksize") or 64)
    absmax_name = stem + ".weight.absmax"
    if absmax_name not in st.tensors:
        raise ValueError(f"{stem}: missing .weight.absmax")
    abs_info = st.tensors[absmax_name]
    abs_raw = st.read_bytes(absmax_name)
    if abs_info.dtype in ("U8", "I8"):
        nq_name = stem + ".weight.nested_quant_map"
        na_name = stem + ".weight.nested_absmax"
        if nq_name not in st.tensors or na_name not in st.tensors:
            raise ValueError(f"{stem}: U8 absmax without nested maps")
        nested_map = unpack_f32(st.read_bytes(nq_name))
        nested_am = unpack_f32(st.read_bytes(na_name))
        nested_bs = int(state.get("nested_blocksize") or 256)
        nested_off = float(state.get("nested_offset") or 0.0)
        absmax = decode_absmax(
            abs_raw,
            absmax_is_u8=True,
            nested_quant_map=nested_map,
            nested_absmax=nested_am,
            nested_offset=nested_off,
            nested_blocksize=nested_bs,
        )
        dq = "double"
    elif abs_info.dtype == "F32":
        absmax = unpack_f32(abs_raw)
        dq = "single"
    else:
        raise ValueError(f"{stem}: absmax dtype {abs_info.dtype} not supported")
    qmap_name = stem + ".weight.quant_map"
    codebook: Optional[List[float]] = None
    src_q = str(state.get("quant_type") or "nf4")
    if qmap_name in st.tensors:
        codebook = unpack_f32(st.read_bytes(qmap_name))
    f32 = dequant_nf4(qweight, absmax, n_elem, blocksize, nibble_order, codebook)
    i8_bs = int8_blocksize if int8_blocksize > 0 else blocksize
    q8, scales = quantize_int8_symmetric(f32, blocksize=i8_bs)
    recon = dequant_int8(q8, scales, blocksize=i8_bs)
    meta = {
        "src": stem,
        "src_quant": src_q,
        "double_quant": dq,
        "shape": [int(shape[0]), int(shape[1])],
        "n_elem": n_elem,
        "nf4_blocksize": blocksize,
        "int8_blocksize": i8_bs,
        "int8_scheme": "symmetric_per_block_zp0",
        "n_scales": len(scales),
        "rmse_vs_src_dequant": rmse(f32, recon),
        "rmse_vs_nf4_dequant": rmse(f32, recon),
        "max_abs_err_vs_src_dequant": max_abs_err(f32, recon),
        "max_abs_err_vs_nf4_dequant": max_abs_err(f32, recon),
    }
    return bytes(q8), scales, meta
def convert_dense_module(
    st: SafeTensorsFile,
    name: str,
    int8_blocksize: int,
) -> Tuple[bytes, List[float], dict]:
    info = st.tensors[name]
    raw = st.read_bytes(name)
    n_elem = numel_of(info.shape)
    i8_bs = int8_blocksize if int8_blocksize > 0 else 64
    used_gpu = False
    f32: Optional[List[float]] = None
    if info.dtype == "BF16" and i8_bs == 64 and n_elem * 2 == len(raw):
        try:
            from gpu_quant import quant_bf16_i8
            got = quant_bf16_i8(raw, blocksize=64)
        except Exception:
            got = None
        if got is not None:
            q8, scales = got
            used_gpu = True
            global _GPU_QUANT_LOGGED
            if not _GPU_QUANT_LOGGED:
                print("INT8_QUANT device=cuda block=64", file=sys.stderr)
                _GPU_QUANT_LOGGED = True
    if not used_gpu:
        f32 = unpack_dense(info.dtype, raw, info.shape)
        if len(f32) < n_elem:
            raise ValueError(f"{name}: dense short {len(f32)} < {n_elem}")
        f32 = f32[:n_elem]
        q8, scales = quantize_int8_symmetric(f32, blocksize=i8_bs)
        q8 = bytes(q8)
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    shape = tuple(int(x) for x in info.shape)
    # Full RMSE on 3B elems is why CPU convert takes ~1h. Sample or skip.
    rmse_v = 0.0
    maxe = 0.0
    if f32 is not None and n_elem <= 65536:
        recon = dequant_int8(q8, scales, blocksize=i8_bs)
        rmse_v = rmse(f32, recon)
        maxe = max_abs_err(f32, recon)
    meta = {
        "src": stem,
        "src_quant": info.dtype.lower(),
        "double_quant": "none",
        "shape": [int(shape[0]), int(shape[1])] if len(shape) == 2 else list(shape),
        "n_elem": n_elem,
        "nf4_blocksize": 0,
        "int8_blocksize": i8_bs,
        "int8_scheme": "symmetric_per_block_zp0",
        "n_scales": len(scales),
        "device": "cuda" if used_gpu else "cpu",
        "rmse_vs_src_dequant": rmse_v,
        "rmse_vs_nf4_dequant": rmse_v,
        "max_abs_err_vs_src_dequant": maxe,
        "max_abs_err_vs_nf4_dequant": maxe,
    }
    return bytes(q8), scales, meta
def _append_int8(
    out_tensors: list,
    stem: str,
    shape: Tuple[int, ...],
    q8: bytes,
    scales: List[float],
    meta: dict,
) -> None:
    out_tensors.append((stem + ".weight", "I8", shape, q8))
    out_tensors.append((stem + ".weight.int8_scale", "F32", (len(scales),), pack_f32(scales)))
    state = {
        "quant_type": "int8_symmetric",
        "blocksize": meta["int8_blocksize"],
        "shape": list(shape),
        "src_quant": meta["src_quant"],
        "double_quant": meta["double_quant"],
    }
    raw_state = json.dumps(state, separators=(",", ":")).encode("utf-8")
    out_tensors.append((stem + ".weight.int8_state", "U8", (len(raw_state),), raw_state))
def convert_dense_to_nf4(
    st: SafeTensorsFile,
    name: str,
    blocksize: int,
) -> Tuple[bytes, List[float], dict]:
    info = st.tensors[name]
    raw = st.read_bytes(name)
    f32 = unpack_dense(info.dtype, raw, info.shape)
    n_elem = numel_of(info.shape)
    f32 = f32[:n_elem]
    packed, absmax = quantize_nf4(f32, blocksize=blocksize)
    recon = dequant_nf4(packed, absmax, n_elem, blocksize)
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    shape = tuple(int(x) for x in info.shape)
    meta = {
        "src": stem,
        "src_quant": info.dtype.lower(),
        "dst_quant": "nf4",
        "double_quant": "none",
        "shape": [int(shape[0]), int(shape[1])] if len(shape) == 2 else list(shape),
        "n_elem": n_elem,
        "nf4_blocksize": blocksize,
        "int8_blocksize": 0,
        "n_scales": len(absmax),
        "rmse_vs_src_dequant": rmse(f32, recon),
        "max_abs_err_vs_src_dequant": max_abs_err(f32, recon),
    }
    return bytes(packed), absmax, meta
def _append_nf4(
    out_tensors: list,
    stem: str,
    shape: Tuple[int, ...],
    packed: bytes,
    absmax: List[float],
    meta: dict,
) -> None:
    out_tensors.append((stem + ".weight", "U8", (len(packed), 1), packed))
    out_tensors.append((stem + ".weight.absmax", "F32", (len(absmax),), pack_f32(absmax)))
    out_tensors.append((stem + ".weight.quant_map", "F32", (16,), pack_f32(list(NF4_CODEBOOK))))
    state = {
        "quant_type": "nf4",
        "blocksize": meta["nf4_blocksize"],
        "dtype": "bfloat16",
        "shape": list(shape),
        "nested_blocksize": 256,
        "nested_dtype": "float32",
        "nested_offset": 0.0,
        "src_quant": meta["src_quant"],
    }
    raw_state = json.dumps(state, separators=(",", ":")).encode("utf-8")
    out_tensors.append(
        (stem + ".weight.quant_state.bitsandbytes__nf4", "U8", (len(raw_state),), raw_state)
    )
def convert_dense_to_nested(
    st: SafeTensorsFile,
    name: str,
    blocksize: int,
) -> Tuple[bytes, bytes, List[float], dict]:
    info = st.tensors[name]
    raw = st.read_bytes(name)
    f32 = unpack_dense(info.dtype, raw, info.shape)
    n_elem = numel_of(info.shape)
    f32 = f32[:n_elem]
    nf4_idx, plug, absmax = encode_nested(f32, blocksize=blocksize)
    hole = bytes(pack_indices(nf4_idx))
    plug_b = bytes(pack_indices(plug))
    recon = decode_nested(nf4_idx, plug, absmax, blocksize)
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    shape = tuple(int(x) for x in info.shape)
    meta = {
        "src": stem,
        "src_quant": info.dtype.lower(),
        "dst_quant": "nested_nf8",
        "shape": [int(shape[0]), int(shape[1])] if len(shape) == 2 else list(shape),
        "n_elem": n_elem,
        "nf4_blocksize": blocksize,
        "n_scales": len(absmax),
        "rmse_vs_src_dequant": rmse(f32, recon),
        "max_abs_err_vs_src_dequant": max_abs_err(f32, recon),
    }
    return hole, plug_b, absmax, meta
def _append_nested(
    out_tensors: list,
    stem: str,
    shape: Tuple[int, ...],
    hole: bytes,
    plug: bytes,
    absmax: List[float],
    meta: dict,
) -> None:
    out_tensors.append((stem + ".weight", "U8", (len(hole), 1), hole))
    out_tensors.append((stem + ".weight.absmax", "F32", (len(absmax),), pack_f32(absmax)))
    out_tensors.append((stem + ".weight.quant_map", "F32", (16,), pack_f32(list(NF4_CODEBOOK))))
    out_tensors.append((stem + ".weight.nf8_plug", "U8", (len(plug), 1), plug))
    state = {
        "quant_type": "nested_nf8",
        "blocksize": meta["nf4_blocksize"],
        "shape": list(shape),
        "src_quant": meta["src_quant"],
        "plug": "4bit_subquantile_in_nf4_cell",
    }
    raw_state = json.dumps(state, separators=(",", ":")).encode("utf-8")
    out_tensors.append((stem + ".weight.nested_state", "U8", (len(raw_state),), raw_state))
def convert_pin(
    src: Path,
    out_dir: Path,
    *,
    int8_blocksize: int = 64,
    dry_run: bool = False,
    policy: Optional[Policy] = None,
) -> dict:
    policy = policy or Policy()
    src = src.expanduser().resolve()
    weight_paths = find_weight_files(src)
    weight_path = weight_paths[0]
    src_dir = src if src.is_dir() else src.parent
    out_dir = out_dir.expanduser().resolve()
    opened = [SafeTensorsFile(str(p)) for p in weight_paths]
    with SafeTensorsSet(opened) as st:
        refused = refuse_reasons(st, src_dir)
        if refused:
            raise ValueError("HARD_BLOCK: " + "; ".join(refused))
        nf4_stems, rest, _consumed = group_modules(st.names())
        if nf4_stems and not policy.allow_requant:
            raise ValueError(
                "HARD_BLOCK: source is already NF4/FP4. Download BF16/F16 and convert once. "
                "Requant (NF4→INT8) needs --allow-requant. That cannot restore lost bits."
            )
        dense_linears = []
        embeds = []
        keep = []
        other = []
        for name in rest:
            info = st.tensors[name]
            if _is_keep(name):
                keep.append(name)
            elif _is_embed(name) and info.dtype in DENSE_DTYPES:
                embeds.append(name)
            elif _is_linear_weight(name, info.shape, info.dtype):
                dense_linears.append(name)
            else:
                other.append(name)
        dest = policy.dest
        if dest not in ("int8", "nf4", "nested-nf8"):
            raise ValueError(f"dest must be int8, nf4, or nested-nf8, got {dest}")
        if nf4_stems and dest == "nested-nf8":
            raise ValueError(
                "HARD_BLOCK: nested-nf8 needs BF16/F16. NF4 has no within-cell plug."
            )
        dense_q = policy.dense in ("int8", "nf4", "quantize")
        embed_q = policy.embed in ("int8", "nf4", "quantize")
        plan = {
            "schema": _dest_schema(dest),
            "src": str(weight_path),
            "n_tensors_in": len(st.tensors),
            "n_nf4_modules": len(nf4_stems),
            "n_dense_linears": len(dense_linears),
            "n_embed": len(embeds),
            "n_keep": len(keep),
            "policy": {
                "dest": dest,
                "allow_requant": policy.allow_requant,
                "dense": "quantize" if dense_q else "copy",
                "embed": "quantize" if embed_q else "copy",
                "norm": "copy",
            },
            "blocksize": int8_blocksize,
            "note": "one hop from BF16/F16/F32. not dest-pack. not orch hot path.",
        }
        if dry_run:
            plan["nf4_modules"] = nf4_stems
            plan["dense_linears"] = dense_linears
            plan["embeds"] = embeds
            plan["keep"] = keep
            plan["other"] = other
            return plan
        out_dir.mkdir(parents=True, exist_ok=True)
        out_tensors: List[Tuple[str, str, Tuple[int, ...], bytes]] = []
        reports: List[dict] = []
        n_copied = 0
        src_kinds = set()
        def emit_quant(name_or_stem: str, is_dense_tensor: bool) -> None:
            nonlocal n_copied
            if is_dense_tensor:
                name = name_or_stem
                stem = name[: -len(".weight")] if name.endswith(".weight") else name
                shape = tuple(int(x) for x in st.tensors[name].shape)
                if dest == "int8":
                    q8, scales, meta = convert_dense_module(st, name, int8_blocksize)
                    _append_int8(out_tensors, stem, shape, q8, scales, meta)
                elif dest == "nested-nf8":
                    hole, plug, absmax, meta = convert_dense_to_nested(
                        st, name, int8_blocksize or 64
                    )
                    _append_nested(out_tensors, stem, shape, hole, plug, absmax, meta)
                else:
                    packed, absmax, meta = convert_dense_to_nf4(st, name, int8_blocksize or 64)
                    _append_nf4(out_tensors, stem, shape, packed, absmax, meta)
                reports.append(meta)
                src_kinds.add(meta["src_quant"])
            else:
                stem = name_or_stem
                if dest == "int8":
                    q8, scales, meta = convert_nf4_module(st, stem, int8_blocksize)
                    _append_int8(out_tensors, stem, tuple(meta["shape"]), q8, scales, meta)
                    reports.append(meta)
                    src_kinds.add(meta["src_quant"])
                else:
                    # already NF4, dest NF4: copy, don't requant
                    w = stem + ".weight"
                    info = st.tensors[w]
                    out_tensors.append((w, info.dtype, info.shape, st.read_bytes(w)))
                    for suf in NF4_SUFFIXES:
                        aux = stem + suf
                        if aux in st.tensors:
                            a = st.tensors[aux]
                            out_tensors.append((aux, a.dtype, a.shape, st.read_bytes(aux)))
                    n_copied += 1
                    src_kinds.add("nf4_copy")
        for stem in nf4_stems:
            emit_quant(stem, False)
        if dense_q:
            for name in dense_linears:
                emit_quant(name, True)
        else:
            for name in dense_linears:
                info = st.tensors[name]
                out_tensors.append((name, info.dtype, info.shape, st.read_bytes(name)))
                n_copied += 1
                src_kinds.add(info.dtype.lower() + "_copy")
        if embed_q:
            for name in embeds:
                emit_quant(name, True)
        else:
            for name in embeds:
                info = st.tensors[name]
                out_tensors.append((name, info.dtype, info.shape, st.read_bytes(name)))
                n_copied += 1
        for name in keep + other:
            info = st.tensors[name]
            out_tensors.append((name, info.dtype, info.shape, st.read_bytes(name)))
            n_copied += 1
        if dest == "nested-nf8":
            out_tensors.append(("_nf8_cells", "F32", (16, 16), pack_f32(cells_flat())))
    if dest == "nf4":
        qtag = "nf4_owned_single_quant"
    elif dest == "nested-nf8":
        qtag = "nested_nf8_gaussian_cell_split"
    else:
        qtag = "int8_symmetric_per_block_zp0"
    write_safetensors(
        str(out_dir / "model.safetensors"),
        out_tensors,
        metadata={
            "format": "pt",
            "quantization": qtag,
            "converted_from": ",".join(sorted(src_kinds)) or "bf16",
        },
    )
    rmses = [m.get("rmse_vs_src_dequant", m.get("rmse_vs_nf4_dequant", 0.0)) for m in reports]
    maxes = [
        m.get("max_abs_err_vs_src_dequant", m.get("max_abs_err_vs_nf4_dequant", 0.0))
        for m in reports
    ]
    pin = {
        "schema": _dest_schema(dest),
        "src": str(src_dir) if len(weight_paths) > 1 else str(weight_path),
        "n_shards": len(weight_paths),
        "dst": str(out_dir / "model.safetensors"),
        "dest": dest,
        "n_nf4_modules": len(nf4_stems),
        "n_dense_modules": len(dense_linears) if dense_q else 0,
        "n_converted": len(reports),
        "n_passthrough": n_copied,
        "src_kinds": sorted(src_kinds),
        "policy": plan["policy"],
        "int8_blocksize": int8_blocksize if dest == "int8" else 0,
        "nf4_blocksize": int8_blocksize if dest in ("nf4", "nested-nf8") else 0,
        "int8_scheme": "symmetric_per_block_zp0" if dest == "int8" else None,
        "rmse_mean": (sum(rmses) / len(rmses)) if rmses else 0.0,
        "max_abs_err_max": max(maxes) if maxes else 0.0,
        "note": "one hop from dense BF16/F16/F32. Ampere train still GEMMs in f16/f32.",
    }
    (out_dir / "pin.json").write_text(json.dumps(pin, indent=2) + "\n")
    (out_dir / "CONVERT_REPORT.json").write_text(
        json.dumps({"pin": pin, "modules": reports}, indent=2) + "\n"
    )
    cfg_path = src_dir / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            cfg = {}
        cfg["quantization_config"] = {
            "quant_method": INT8_PIN_QUANT_METHOD,
            "quant_method_legacy": "nf4_to_int8_pin",
            "load_in_8bit": True,
            "int8_scheme": "symmetric_per_block_zp0",
            "int8_blocksize": int8_blocksize,
            "converted_from": sorted(src_kinds),
        }
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name in COPY_SIDE_FILES:
        if name == "config.json":
            continue
        src_f = src_dir / name
        if src_f.is_file():
            shutil.copy2(src_f, out_dir / name)
    return {"pin": pin, "modules": reports}
def write_tiny_fixture(out_dir: Path) -> dict:
    """8×64 single-quant NF4 → INT8 pin. For the orch loader CI. Not Phi-4."""
    from nf4_to_int8 import demo_weights
    out_dir = out_dir.expanduser().resolve()
    out_f, in_f = 8, 64
    w = demo_weights(out_f * in_f)
    qw, am = quantize_nf4(w, blocksize=64)
    state = json.dumps(
        {
            "quant_type": "nf4",
            "blocksize": 64,
            "dtype": "bfloat16",
            "shape": [out_f, in_f],
            "nested_blocksize": 256,
            "nested_dtype": "float32",
            "nested_offset": 0.0,
        },
        separators=(",", ":"),
    ).encode()
    src = out_dir / "_src"
    src.mkdir(parents=True, exist_ok=True)
    write_safetensors(
        str(src / "model.safetensors"),
        [
            ("model.layers.0.mlp.down_proj.weight", "U8", (len(qw), 1), bytes(qw)),
            ("model.layers.0.mlp.down_proj.weight.absmax", "F32", (len(am),), pack_f32(am)),
            ("model.layers.0.mlp.down_proj.weight.quant_map", "F32", (16,), pack_f32(list(NF4_CODEBOOK))),
            (
                "model.layers.0.mlp.down_proj.weight.quant_state.bitsandbytes__nf4",
                "U8",
                (len(state),),
                state,
            ),
            ("model.norm.weight", "BF16", (8,), b"\x00\x3c" * 8),
        ],
        metadata={"format": "pt"},
    )
    (src / "config.json").write_text(json.dumps({"model_type": "phi3", "hidden_size": 8}) + "\n")
    return convert_pin(
        src,
        out_dir / "int8_pin",
        int8_blocksize=64,
        policy=Policy(dest="int8", allow_requant=True, dense="quantize"),
    )
def cmd_pin(args: Any) -> int:
    src = Path(args.src)
    out = Path(args.out)
    dense = getattr(args, "dense", "quantize")
    if dense == "int8":
        dense = "quantize"
    embed = getattr(args, "embed", "copy")
    if embed == "int8":
        embed = "quantize"
    policy = Policy(
        dest=getattr(args, "to", "int8"),
        allow_requant=bool(getattr(args, "allow_requant", False)),
        dense=dense,
        embed=embed,
    )
    got = convert_pin(
        src,
        out,
        int8_blocksize=args.int8_blocksize,
        dry_run=args.dry_run,
        policy=policy,
    )
    print(json.dumps(got if args.dry_run else got["pin"], indent=2))
    return 0
