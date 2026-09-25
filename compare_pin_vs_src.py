#!/usr/bin/env python3
"""Compare an INT8 pin to its original dense BF16/F16 source.
Two residuals, kept separate:
1. Transfer: re-quant the source with the same GPU kernel and count I8/scale
   mismatches. Zero means the pin bytes are the converter output.
2. Scheme fuzz: dequant (i8 * per-block scale) vs original BF16. Bound is
   half the block scale (round-to-nearest, zp=0, clip ±127).
Carve-out. Not orch train.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from nf4 import unpack_f32
from pin_convert import find_weight_files
from safetensors_io import SafeTensorsFile, SafeTensorsSet
from gpu_quant import compare_bf16_i8, gpu_quant_available, quant_bf16_i8
def _open_set(path: Path) -> SafeTensorsSet:
    files = [SafeTensorsFile(str(p)) for p in find_weight_files(path)]
    return SafeTensorsSet(files)
def _scale_rel_max(a: List[float], b: List[float]) -> float:
    m = 0.0
    n = min(len(a), len(b))
    for i in range(n):
        den = max(abs(a[i]), abs(b[i]), 1e-30)
        r = abs(a[i] - b[i]) / den
        if r > m:
            m = r
    return m
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", required=True, help="INT8 pin directory")
    ap.add_argument("--src", default="", help="override BF16 snapshot dir")
    ap.add_argument("--out", default="", help="JSON report path")
    args = ap.parse_args()
    pin_dir = Path(args.pin)
    pin_json_path = pin_dir / "pin.json"
    if not pin_json_path.is_file():
        print(f"HARD_BLOCK: missing {pin_json_path}", file=sys.stderr)
        return 2
    pin_meta = json.loads(pin_json_path.read_text())
    src_dir = Path(args.src) if args.src else Path(pin_meta.get("src") or "")
    if not src_dir.exists():
        print(f"HARD_BLOCK: missing src {src_dir}", file=sys.stderr)
        return 2
    if not gpu_quant_available():
        print("HARD_BLOCK: libquant_i8.so / CUDA compare unavailable", file=sys.stderr)
        return 2
    t0 = time.time()
    pin_st = _open_set(pin_dir)
    src_st = _open_set(src_dir)
    linears: List[Dict[str, Any]] = []
    passthrough: List[Dict[str, Any]] = []
    total_ss = 0.0
    total_n = 0
    global_max = 0.0
    global_over = 0
    global_i8_mis = 0
    global_scale_mis = 0
    n_linears = 0
    n_pass = 0
    n_pass_mis = 0
    names = sorted(pin_st.names())
    for name in names:
        info = pin_st.tensors[name]
        if name.endswith(".weight.int8_scale") or name.endswith(".weight.int8_state"):
            continue
        if info.dtype == "I8" and name.endswith(".weight"):
            scale_name = name + ".int8_scale"
            if scale_name not in pin_st.tensors:
                print(f"HARD_BLOCK: I8 without scale: {name}", file=sys.stderr)
                return 2
            if name not in src_st.tensors:
                print(f"HARD_BLOCK: src missing {name}", file=sys.stderr)
                return 2
            src_info = src_st.tensors[name]
            if src_info.dtype != "BF16":
                print(
                    f"HARD_BLOCK: src {name} dtype={src_info.dtype} (want BF16)",
                    file=sys.stderr,
                )
                return 2
            src_raw = src_st.read_bytes(name)
            q8 = pin_st.read_bytes(name)
            sc_raw = pin_st.read_bytes(scale_name)
            scales = unpack_f32(sc_raw)
            n = len(q8)
            if len(src_raw) != n * 2:
                print(
                    f"HARD_BLOCK: size mismatch {name} src={len(src_raw)} i8={n}",
                    file=sys.stderr,
                )
                return 2
            fresh = quant_bf16_i8(src_raw, blocksize=64)
            if fresh is None:
                print(f"HARD_BLOCK: GPU requant failed {name}", file=sys.stderr)
                return 2
            q_fresh, sc_fresh = fresh
            i8_mis = 0
            if q_fresh != q8:
                i8_mis = sum(1 for a, b in zip(q_fresh, q8) if a != b)
            sc_rel = _scale_rel_max(scales, sc_fresh)
            sc_mis = 0 if sc_rel <= 1e-6 else 1
            err = compare_bf16_i8(src_raw, q8, scales, blocksize=64)
            if err is None:
                print(f"HARD_BLOCK: GPU compare failed {name}", file=sys.stderr)
                return 2
            n_linears += 1
            total_ss += err["sum_sq"]
            total_n += err["n"]
            if err["max_abs"] > global_max:
                global_max = err["max_abs"]
            global_over += err["n_over_half_scale"]
            global_i8_mis += i8_mis
            global_scale_mis += sc_mis
            max_scale = max(abs(x) for x in scales) if scales else 0.0
            half = 0.5 * max_scale
            rec = {
                "name": name,
                "n_elem": n,
                "shape": list(info.shape),
                "i8_byte_mismatch": i8_mis,
                "scale_rel_max": sc_rel,
                "rmse_vs_bf16": err["rmse"],
                "max_abs_vs_bf16": err["max_abs"],
                "n_over_half_scale": err["n_over_half_scale"],
                "max_block_scale": max_scale,
                "half_lsb_at_max_scale": half,
                "max_abs_over_half_lsb": (err["max_abs"] / half) if half > 0 else 0.0,
            }
            linears.append(rec)
            if n_linears % 8 == 0 or n_linears == 1:
                print(
                    f"linear {n_linears} {name} rmse={err['rmse']:.6g} "
                    f"maxabs={err['max_abs']:.6g} i8_mis={i8_mis} "
                    f"over={err['n_over_half_scale']}",
                    file=sys.stderr,
                    flush=True,
                )
            continue
        if info.dtype in ("BF16", "F16", "F32"):
            n_pass += 1
            src_ok = name in src_st.tensors
            mis = 1
            n_bytes = info.nbytes
            if src_ok:
                a = pin_st.read_bytes(name)
                b = src_st.read_bytes(name)
                mis = 0 if a == b else 1
                n_bytes = len(a)
            n_pass_mis += mis
            passthrough.append(
                {
                    "name": name,
                    "dtype": info.dtype,
                    "n_bytes": n_bytes,
                    "in_src": src_ok,
                    "byte_mismatch": mis,
                }
            )
    pin_st.close()
    src_st.close()
    rmse = math.sqrt(total_ss / float(total_n)) if total_n else 0.0
    transfer_exact = (
        global_i8_mis == 0 and global_scale_mis == 0 and n_pass_mis == 0
    )
    scheme_within_half = global_over == 0
    worst = sorted(linears, key=lambda r: r["max_abs_vs_bf16"], reverse=True)[:8]
    worst_rmse = sorted(linears, key=lambda r: r["rmse_vs_bf16"], reverse=True)[:8]
    report = {
        "schema": "int8_pin_vs_bf16_v1",
        "pin": str(pin_dir),
        "src": str(src_dir),
        "src_kind": (pin_meta.get("src_kinds") or [pin_meta.get("src_kind")])[0],
        "n_linears": n_linears,
        "n_passthrough": n_pass,
        "n_elem_linears": total_n,
        "transfer_exact": transfer_exact,
        "i8_byte_mismatch_total": global_i8_mis,
        "scale_mismatch_modules": global_scale_mis,
        "passthrough_mismatch_tensors": n_pass_mis,
        "scheme": "symmetric_per_block_zp0",
        "blocksize": 64,
        "rmse_vs_bf16": rmse,
        "max_abs_vs_bf16": global_max,
        "n_over_half_scale": global_over,
        "scheme_within_half_lsb": scheme_within_half,
        "worst_max_abs": worst,
        "worst_rmse": worst_rmse,
        "seconds": time.time() - t0,
        "note": (
            "pin.json rmse_mean=0.0 was skipped large-tensor RMSE, not lossless. "
            "This report is dequant vs original BF16 plus re-quant byte match."
        ),
    }
    text = json.dumps(report, indent=2)
    out = Path(args.out) if args.out else pin_dir / "COMPARE_VS_BF16.json"
    out.write_text(text + "\n")
    print(
        f"COMPARE_VS_BF16 transfer_exact={str(transfer_exact).lower()} "
        f"scheme_within_half_lsb={str(scheme_within_half).lower()} "
        f"rmse={rmse:.8g} max_abs={global_max:.8g} n_over={global_over} "
        f"i8_mis={global_i8_mis} pass_mis={n_pass_mis} "
        f"n_lin={n_linears} n_elem={total_n} out={out}",
        flush=True,
    )
    return 0 if transfer_exact and scheme_within_half else 3
if __name__ == "__main__":
    raise SystemExit(main())
