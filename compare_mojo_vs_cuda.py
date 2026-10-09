#!/usr/bin/env python3
"""Twin check: Mojo INT8 quantizer (mojo/quant_i8_mojo.so) vs CUDA (cuda/libquant_i8.so).

Same BF16 input through every available path: cuda, mojo-gpu, mojo-cpu. Per path:
max abs dequant error vs the BF16 source, elements over half their own block scale, codes and
scales that differ from the CUDA path, codes that differ from the Python golden on
a prefix, and median wall time per call (H2D + kernel + D2H, end to end).

Build first: bash cuda/build_quant_i8.sh && bash mojo/build.sh. Not train.
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
import struct
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dtype_io import unpack_bf16
from nf4 import dequant_int8, max_abs_err, quantize_int8_symmetric
import gpu_quant
import mojo_quant

B = 64
QuantOut = Optional[Tuple[bytes, List[float]]]


def make_bf16(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    f = struct.pack(f"<{n}f", *(rng.gauss(0.0, 0.02) for _ in range(n)))
    # High half of each f32 = truncated BF16 (as dtype_io.pack_bf16).
    out = bytearray(n * 2)
    out[0::2] = f[2::4]
    out[1::2] = f[3::4]
    # A few outliers so some blocks have a much larger scale.
    for i in range(0, n, 4099):
        out[2 * i : 2 * i + 2] = struct.pack("<f", -1.5 if i % 2 else 2.0)[2:]
    return bytes(out)


def max_err(raw: bytes, q: bytes, sc: List[float]) -> Tuple[float, int]:
    """(max |w - q*s|, elements over half their own block's scale).

    GPU reduce when CUDA is built; the CPU fallback uses the same per-block
    slack as k_err_bf16 in cuda/quant_i8.cu.
    """
    r = gpu_quant.compare_bf16_i8(raw, q, sc, B)
    if r is not None:
        return r["max_abs"], r["n_over_half_scale"]
    vals = unpack_bf16(raw)
    rec = dequant_int8(q, sc, B)
    over = 0
    for i, (w, d) in enumerate(zip(vals, rec)):
        half = 0.5 * sc[i // B]
        if abs(w - d) > half * 1.0000002 + 1e-6:
            over += 1
    return max_abs_err(vals, rec), over


def timed(fn: Callable[[], QuantOut], reps: int) -> Tuple[QuantOut, float]:
    out = fn()  # warmup (driver / context init)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, statistics.median(ts)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=1 << 22, help="BF16 elements (default 4M)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--golden-n", type=int, default=1 << 16, help="prefix checked vs nf4 golden")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    n = a.n
    raw = make_bf16(n, a.seed)

    paths: Dict[str, Callable[[], QuantOut]] = {}
    if gpu_quant.gpu_quant_available():
        paths["cuda"] = lambda: gpu_quant.quant_bf16_i8(raw, B)
    if mojo_quant.mojo_quant_available():
        paths["mojo-gpu"] = lambda: mojo_quant.quant_bf16_i8(raw, B, device="gpu")
        paths["mojo-cpu"] = lambda: mojo_quant.quant_bf16_i8(raw, B, device="cpu")
    if not paths:
        print("no quant path built (cuda/libquant_i8.so, mojo/quant_i8_mojo.so)", file=sys.stderr)
        return 2

    gn = min(a.golden_n, n)
    # Quantize whole blocks so a gn that is not a multiple of B sees the same
    # last-block scale as the kernels; compare codes on the first gn only.
    gend = min(n, -(-gn // B) * B)
    gq, gsc = quantize_int8_symmetric(unpack_bf16(raw[: 2 * gend]), B)
    gq = gq[:gn]
    ref: Optional[Tuple[bytes, List[float]]] = None
    report: Dict[str, Any] = {"n": n, "blocksize": B, "reps": a.reps, "golden_n": gn, "paths": {}}
    for name, fn in paths.items():
        out, sec = timed(fn, a.reps)
        if out is None:
            report["paths"][name] = {"ok": False}
            continue
        q, sc = out
        err, n_over = max_err(raw, q, sc)
        row: Dict[str, Any] = {
            "ok": True,
            "max_abs_err": err,
            "n_over_half_scale": n_over,
            "within_bound": n_over == 0,
            "golden_code_mismatch": sum(1 for x, y in zip(q[:gn], gq) if x != y),
            "golden_scale_rel_max": max(abs(s - g) / g for s, g in zip(sc, gsc)),
            "median_s": sec,
            "gelem_per_s": n / sec / 1e9,
        }
        if ref is None:
            ref = out
            report["ref"] = name
        else:
            row["code_mismatch_vs_ref"] = sum(1 for x, y in zip(q, ref[0]) if x != y)
            row["scale_mismatch_vs_ref"] = sum(1 for x, y in zip(sc, ref[1]) if x != y)
        report["paths"][name] = row
    print(json.dumps(report, indent=2))
    bad = [k for k, v in report["paths"].items()
           if not v["ok"] or not v["within_bound"]
           or v.get("code_mismatch_vs_ref", 0) or v.get("scale_mismatch_vs_ref", 0)]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
