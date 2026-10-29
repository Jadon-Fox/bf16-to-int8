"""Optional numpy twins of the stdlib quant loops in nf4.py / nested_nf.py.
Same math, same bytes: f64 host arithmetic, round-half-even, first-min ties.
None of this is required. Set BF16_TO_INT8_NO_NUMPY=1 to force the stdlib path.
"""
from __future__ import annotations
import os
from typing import Optional, Sequence, Tuple

try:
    if os.environ.get("BF16_TO_INT8_NO_NUMPY"):
        raise ImportError("disabled by BF16_TO_INT8_NO_NUMPY")
    import numpy as np
except ImportError:  # stdlib-only install
    np = None  # type: ignore[assignment]

from nf4 import NF4_CODEBOOK, NIBBLE_LO_THEN_HI


def available() -> bool:
    return np is not None


def unpack_dense(dtype: str, raw: bytes, n_elem: int) -> "np.ndarray":
    """BF16 / F16 / F32 little-endian → f64 (exact widening, like dtype_io)."""
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype="<u2", count=n_elem).astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float64)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2", count=n_elem).astype(np.float64)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4", count=n_elem).astype(np.float64)
    raise ValueError(f"dense unpack does not support dtype {dtype}")


def _blocks(x: "np.ndarray", blocksize: int) -> "np.ndarray":
    """[n_blocks, blocksize] view, zero-padded tail (|0| never raises a block max)."""
    n = x.shape[0]
    nb = (n + blocksize - 1) // blocksize
    pad = nb * blocksize - n
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    return x.reshape(nb, blocksize)


def block_absmax(x: "np.ndarray", blocksize: int) -> "np.ndarray":
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.abs(_blocks(x, blocksize)).max(axis=1)


def quantize_int8(x: "np.ndarray", blocksize: int = 64) -> Tuple["np.ndarray", "np.ndarray"]:
    """q = clip(rint(w / s), -127, 127), s = amax/127 (1.0 for an all-zero block)."""
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.float64)
    xb = _blocks(x, blocksize)
    am = np.abs(xb).max(axis=1)
    scales = np.where(am > 0.0, am / 127.0, 1.0)
    t = xb / scales[:, None]
    np.rint(t, out=t)
    np.clip(t, -127, 127, out=t)
    return t.astype(np.int8).reshape(-1)[:n], scales


def dequant_int8(q: "np.ndarray", scales: "np.ndarray", blocksize: int = 64) -> "np.ndarray":
    n = q.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    qb = _blocks(q, blocksize).astype(np.float64)
    qb *= np.asarray(scales, dtype=np.float64)[:, None]
    return qb.reshape(-1)[:n]


def _nearest(v: "np.ndarray", table: "np.ndarray") -> "np.ndarray":
    """Index of nearest entry, first wins on ties (matches nf4.nearest_code)."""
    return np.abs(v[:, None] - table[None, :]).argmin(axis=1)


def _chunks(n: int, step: int = 1 << 20):
    for lo in range(0, n, step):
        yield lo, min(n, lo + step)


def pack_nibbles(idx: "np.ndarray", nibble_order: int = NIBBLE_LO_THEN_HI) -> bytes:
    idx = idx.astype(np.uint8) & 0x0F
    if idx.shape[0] & 1:
        idx = np.concatenate([idx, np.zeros(1, dtype=np.uint8)])
    first, second = idx[0::2], idx[1::2]
    if nibble_order == NIBBLE_LO_THEN_HI:
        return (first | (second << 4)).astype(np.uint8).tobytes()
    return (second | (first << 4)).astype(np.uint8).tobytes()


def unpack_nibbles(packed: bytes, n_elem: int, nibble_order: int = NIBBLE_LO_THEN_HI) -> "np.ndarray":
    b = np.frombuffer(packed, dtype=np.uint8, count=(n_elem + 1) // 2)
    lo, hi = b & 0x0F, b >> 4
    out = np.empty(b.shape[0] * 2, dtype=np.uint8)
    if nibble_order == NIBBLE_LO_THEN_HI:
        out[0::2], out[1::2] = lo, hi
    else:
        out[0::2], out[1::2] = hi, lo
    return out[:n_elem]


def quantize_nf4(
    x: "np.ndarray", blocksize: int = 64, nibble_order: int = NIBBLE_LO_THEN_HI
) -> Tuple[bytes, "np.ndarray"]:
    am = block_absmax(x, blocksize)
    s = np.repeat(am, blocksize)[: x.shape[0]]
    cb = np.asarray(NF4_CODEBOOK, dtype=np.float64)
    idx = np.full(x.shape[0], 7, dtype=np.uint8)
    for lo, hi in _chunks(x.shape[0]):
        ss = s[lo:hi]
        nz = ss != 0.0
        part = np.full(hi - lo, 7, dtype=np.uint8)
        part[nz] = _nearest(x[lo:hi][nz] / ss[nz], cb)
        idx[lo:hi] = part
    return pack_nibbles(idx, nibble_order), am


def dequant_nf4(
    packed: bytes,
    absmax: Sequence[float],
    n_elem: int,
    blocksize: int = 64,
    nibble_order: int = NIBBLE_LO_THEN_HI,
    codebook: Optional[Sequence[float]] = None,
) -> "np.ndarray":
    cb = np.asarray(codebook if codebook is not None else NF4_CODEBOOK, dtype=np.float64)
    if cb.shape[0] != 16:
        raise ValueError("4-bit codebook must have 16 levels")
    need_a = (n_elem + blocksize - 1) // blocksize
    am = np.asarray(absmax, dtype=np.float64)
    if len(packed) < (n_elem + 1) // 2:
        raise ValueError(f"qweight short: have {len(packed)} need {(n_elem + 1) // 2}")
    if am.shape[0] < need_a:
        raise ValueError(f"absmax short: have {am.shape[0]} need {need_a}")
    idx = unpack_nibbles(packed, n_elem, nibble_order)
    return cb[idx] * np.repeat(am[:need_a], blocksize)[:n_elem]


def encode_nested(
    x: "np.ndarray", cells: Sequence[Sequence[float]], blocksize: int = 64
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """nf4 idx, plug, absmax. Twin of nested_nf.encode_nested."""
    am = block_absmax(x, blocksize)
    s = np.repeat(np.where(am != 0.0, am, 1.0), blocksize)[: x.shape[0]]
    cb = np.asarray(NF4_CODEBOOK, dtype=np.float64)
    cl = np.asarray(cells, dtype=np.float64)
    nf4 = np.empty(x.shape[0], dtype=np.uint8)
    plug = np.empty(x.shape[0], dtype=np.uint8)
    for lo, hi in _chunks(x.shape[0]):
        v = x[lo:hi] / s[lo:hi]
        k = _nearest(v, cb)
        sub = cl[k]  # [m, 16]
        nf4[lo:hi] = k
        plug[lo:hi] = np.abs(v[:, None] - sub).argmin(axis=1)
    return nf4, plug, am


def decode_nested(
    nf4: "np.ndarray", plug: "np.ndarray", absmax: Sequence[float],
    cells: Sequence[Sequence[float]], blocksize: int = 64,
) -> "np.ndarray":
    cl = np.asarray(cells, dtype=np.float64)
    am = np.asarray(absmax, dtype=np.float64)
    return cl[nf4, plug] * np.repeat(am, blocksize)[: nf4.shape[0]]


def err_stats(a: "np.ndarray", b: "np.ndarray") -> Tuple[float, float]:
    """(rmse, max_abs) of a - b."""
    if a.shape[0] == 0:
        return 0.0, 0.0
    d = a - b
    return float(np.sqrt(np.mean(d * d))), float(np.abs(d).max())


def pack_f32(x: "np.ndarray") -> bytes:
    return np.asarray(x, dtype="<f4").tobytes()
