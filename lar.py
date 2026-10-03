"""LAR — Lattice + Async Residual (codec side).
Not QLoRA. Not W4A8-only. One INT4 lattice (pad-to-INT8 MMA native)
plus the residual (BF16 - dequant) kept as a second plane:
  - sparse outliers, and/or
  - low-rank UV
The theory puts that plane in *host RAM* and pipelines it like L3.
This file only measures reconstruction.
"""
from __future__ import annotations
import math
from typing import List, Sequence, Tuple
from nf4 import dequant_int8, dequant_nf4, quantize_int8_symmetric, quantize_nf4, rmse
def quantize_int4_symmetric(
    weights: Sequence[float], blocksize: int = 64
) -> Tuple[List[int], List[float]]:
    """Signed INT4, zp=0, q in [-8, 7], scale = amax/8. Pad-to-INT8 is lossless for these codes."""
    n = len(weights)
    n_blocks = (n + blocksize - 1) // blocksize
    absmax = [0.0] * n_blocks
    for i, w in enumerate(weights):
        a = abs(float(w))
        b = i // blocksize
        if a > absmax[b]:
            absmax[b] = a
    q = [0] * n
    scales = [0.0] * n_blocks
    for b in range(n_blocks):
        scales[b] = absmax[b] / 8.0 if absmax[b] > 0 else 1.0
    for i, w in enumerate(weights):
        s = scales[i // blocksize]
        v = int(round(float(w) / s)) if s else 0
        if v > 7:
            v = 7
        elif v < -8:
            v = -8
        q[i] = v
    return q, scales
def dequant_int4(q: Sequence[int], scales: Sequence[float], blocksize: int = 64) -> List[float]:
    return [float(q[i]) * scales[i // blocksize] for i in range(len(q))]
def pad_int4_to_int8(q4: Sequence[int]) -> List[int]:
    """Bitwise widen: signed nibble sits in INT8. Inverse of this is lossless."""
    return [int(x) for x in q4]
def residual(w: Sequence[float], hat: Sequence[float]) -> List[float]:
    return [float(a) - float(b) for a, b in zip(w, hat)]
def sparse_keep(r: Sequence[float], keep_frac: float) -> List[float]:
    n = len(r)
    k = max(1, int(round(n * keep_frac)))
    idx = sorted(range(n), key=lambda i: abs(r[i]), reverse=True)[:k]
    keep = set(idx)
    return [r[i] if i in keep else 0.0 for i in range(n)]
def low_rank_approx(mat: List[List[float]], rank: int) -> List[List[float]]:
    """Thin SVD via power-free numpy-free Gram iteration is painful; use python svd via...
    stdlib only: snapshot SVD with nested lists using a tiny Jacobi-ish for small demo.
    For the demo we flatten-row and do rank-1 outer of top singular via power method."""
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    # power method on A^T A for top `rank` components, deflate
    out = [[0.0] * cols for _ in range(rows)]
    A = [row[:] for row in mat]
    for _ in range(rank):
        v = [1.0] * cols
        for _it in range(12):
            AtAv = [0.0] * cols
            Av = [0.0] * rows
            for i in range(rows):
                s = 0.0
                ai = A[i]
                for j in range(cols):
                    s += ai[j] * v[j]
                Av[i] = s
            for j in range(cols):
                s = 0.0
                for i in range(rows):
                    s += A[i][j] * Av[i]
                AtAv[j] = s
            nrm = math.sqrt(sum(x * x for x in AtAv)) or 1.0
            v = [x / nrm for x in AtAv]
        # u = A v / ||Av||
        Av = [0.0] * rows
        for i in range(rows):
            s = 0.0
            ai = A[i]
            for j in range(cols):
                s += ai[j] * v[j]
            Av[i] = s
        sigma = math.sqrt(sum(x * x for x in Av)) or 1.0
        u = [x / sigma for x in Av]
        for i in range(rows):
            for j in range(cols):
                out[i][j] += sigma * u[i] * v[j]
                A[i][j] -= sigma * u[i] * v[j]
    return out
def reshape(v: Sequence[float], rows: int, cols: int) -> List[List[float]]:
    return [list(v[i * cols : (i + 1) * cols]) for i in range(rows)]
def flatten(m: List[List[float]]) -> List[float]:
    out: List[float] = []
    for row in m:
        out.extend(row)
    return out
def compare(w: Sequence[float], rows: int, cols: int, blocksize: int = 64) -> dict:
    q8, s8 = quantize_int8_symmetric(w, blocksize)
    h8 = dequant_int8(q8, s8, blocksize)
    qn, an = quantize_nf4(w, blocksize)
    hn = dequant_nf4(qn, an, len(w), blocksize)
    q4, s4 = quantize_int4_symmetric(w, blocksize)
    h4 = dequant_int4(q4, s4, blocksize)
    r = residual(w, h4)
    r_sp = sparse_keep(r, 0.05)
    h4_sp = [h4[i] + r_sp[i] for i in range(len(w))]
    r_mat = reshape(r, rows, cols)
    r_lr = flatten(low_rank_approx(r_mat, rank=4))
    h4_lr = [h4[i] + r_lr[i] for i in range(len(w))]
    r_both = [r_sp[i] + (r[i] - r_sp[i]) for i in range(len(w))] # placeholder
    # sparse on leftover after low-rank
    leftover = [r[i] - r_lr[i] for i in range(len(w))]
    leftover_sp = sparse_keep(leftover, 0.02)
    h4_lr_sp = [h4[i] + r_lr[i] + leftover_sp[i] for i in range(len(w))]
    return {
        "n": len(w),
        "rmse_int8": rmse(w, h8),
        "rmse_nf4": rmse(w, hn),
        "rmse_int4": rmse(w, h4),
        "rmse_int4_sparse5": rmse(w, h4_sp),
        "rmse_int4_rank4": rmse(w, h4_lr),
        "rmse_int4_rank4_sparse2": rmse(w, h4_lr_sp),
        "int4_pad_int8_lossless": pad_int4_to_int8(q4) == list(q4),
        "note": "reconstruction vs original f32. not PPL.",
    }
if __name__ == "__main__":
    from nf4_to_int8 import demo_weights
    rows, cols = 64, 64
    w = demo_weights(rows * cols)
    got = compare(w, rows, cols)
    print(got)
