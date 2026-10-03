"""Nested NormalFloat: split one NF4 Gaussian cell into 16 sub-cells → 256 (NF8).
Hole = existing NF4 nibble (H-TILE qpack).
Plug = which of 16 sub-quantiles inside that cell (4 bits).
Inflate = NF8 reconstruction = information-theoretically the 8-bit
          Gaussian scalar code, *nested* on the 16 buckets, not uniform INT8.
Lossless vs NF8 if the plug is stored. Not bit-exact BF16
(that leftover is a third plane).
"""
from __future__ import annotations
import math
from typing import List, Sequence, Tuple
from nf4 import (
    NF4_CODEBOOK,
    dequant_int8,
    dequant_nf4,
    pack_f32,
    pack_indices,
    quantize_int8_symmetric,
    quantize_nf4,
    rmse,
    unpack_indices,
)
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _norm_ppf(p: float) -> float:
    """Acklam inverse-normal. p in (0,1)."""
    if p <= 0.0:
        return -1e9
    if p >= 1.0:
        return 1e9
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577459574091e02,
        -3.066479806614736e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
def voronoi_bounds(cb: Sequence[float]) -> List[float]:
    b = [float(cb[0])]
    for i in range(len(cb) - 1):
        b.append(0.5 * (float(cb[i]) + float(cb[i + 1])))
    b[0] = -1.0
    b.append(1.0)
    return b
def subdivide_cell(lo: float, hi: float, n: int = 16) -> List[float]:
    """n reconstruction points = equal Gaussian *mass* in (lo,hi) ∩ support.
    Cells are on the *normalized* axis (the NF4 codebook domain)."""
    # Map codebook domain [-1,1] ~ truncated normal via the NF4 endpoints.
    # Equal mass in CDF space between cdf(lo_z) and cdf(hi_z), then ppf back.
    # Use the codebook values as already-normalized z/zmax; treat as z in R
    # after atanh-ish: just ppf-interp in probability of a N(0,s) on R,
    # with lo,hi as z values scaled so |z|<=~1 corresponds to the table.
    zlo, zhi = lo, hi
    p0 = _norm_cdf(zlo * 3.0) # NF4 table spans ~[-1,1] ~ ±3σ after absmax
    p1 = _norm_cdf(zhi * 3.0)
    if p1 <= p0:
        p1 = p0 + 1e-12
    out = []
    for i in range(n):
        p = p0 + (p1 - p0) * (i + 0.5) / n
        z = _norm_ppf(p) / 3.0
        if z < lo:
            z = lo
        if z > hi:
            z = hi
        out.append(z)
    return out
def nf8_nested_from_nf4() -> List[List[float]]:
    bounds = voronoi_bounds(NF4_CODEBOOK)
    cells = []
    for k in range(16):
        cells.append(subdivide_cell(bounds[k], bounds[k + 1], 16))
    return cells
NF8_CELLS = nf8_nested_from_nf4()
def encode_nested(
    weights: Sequence[float], blocksize: int = 64
) -> Tuple[List[int], List[int], List[float]]:
    """Returns nf4_idx, plug 0..15, absmax. Plug picks the sub-bucket."""
    n = len(weights)
    n_blocks = (n + blocksize - 1) // blocksize
    absmax = [0.0] * n_blocks
    for i, w in enumerate(weights):
        a = abs(float(w))
        b = i // blocksize
        if a > absmax[b]:
            absmax[b] = a
    nf4 = [0] * n
    plug = [0] * n
    cells = NF8_CELLS
    for i, w in enumerate(weights):
        s = absmax[i // blocksize] or 1.0
        v = float(w) / s
        best_k, best_d = 0, abs(v - NF4_CODEBOOK[0])
        for k in range(1, 16):
            d = abs(v - NF4_CODEBOOK[k])
            if d < best_d:
                best_d, best_k = d, k
        nf4[i] = best_k
        sub = cells[best_k]
        best_j, best_d = 0, abs(v - sub[0])
        for j in range(1, 16):
            d = abs(v - sub[j])
            if d < best_d:
                best_d, best_j = d, j
        plug[i] = best_j
    return nf4, plug, absmax
def decode_nested(
    nf4: Sequence[int], plug: Sequence[int], absmax: Sequence[float], blocksize: int = 64
) -> List[float]:
    cells = NF8_CELLS
    out = [0.0] * len(nf4)
    for i in range(len(nf4)):
        s = absmax[i // blocksize]
        out[i] = cells[nf4[i]][plug[i]] * s
    return out
def compare(w: Sequence[float], blocksize: int = 64) -> dict:
    qn, an = quantize_nf4(w, blocksize)
    hn = dequant_nf4(qn, an, len(w), blocksize)
    q8, s8 = quantize_int8_symmetric(w, blocksize)
    h8 = dequant_int8(q8, s8, blocksize)
    nf4, plug, am = encode_nested(w, blocksize)
    hnest = decode_nested(nf4, plug, am, blocksize)
    return {
        "rmse_nf4": rmse(w, hn),
        "rmse_uniform_int8": rmse(w, h8),
        "rmse_nested_nf8": rmse(w, hnest),
        "plug_is_4bit": all(0 <= p < 16 for p in plug),
        "note": "nested NF8 = split each of 16 Gaussian cells into 16. not IEEE LSBs. not ",
    }
def cells_flat() -> List[float]:
    out: List[float] = []
    for row in NF8_CELLS:
        out.extend(row)
    return out
def write_c_header(path: str) -> None:
    lines = [
        "/* Generated by nested_nf.py. Nested NF8: NF8_CELLS[nf4][plug]. */",
        "#ifndef ORCH_NF8_CELLS_H_",
        "#define ORCH_NF8_CELLS_H_",
        "",
        "static const float NF8_CELLS[16][16] = {",
    ]
    for row in NF8_CELLS:
        body = ", ".join(f"{x:.9e}f" for x in row)
        lines.append(f" {{ {body} }},")
    lines += ["};", "", "#endif /* ORCH_NF8_CELLS_H_ */", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
if __name__ == "__main__":
    import random
    import sys
    from pathlib import Path
    if len(sys.argv) >= 3 and sys.argv[1] == "--header":
        write_c_header(sys.argv[2])
        print(sys.argv[2])
        raise SystemExit(0)
    random.seed(0)
    random.seed(0)
    w = [random.gauss(0, 0.02) for _ in range(4096)]
    print(compare(w))
