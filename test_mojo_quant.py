#!/usr/bin/env python3
"""CPU golden checks for the Mojo INT8 twin (mojo/quant_i8.mojo) vs nf4.quantize_int8_symmetric.

Skipped when mojo/quant_i8_mojo.so is not built (`bash mojo/build.sh`). Not train.

The golden divides in float64; the Mojo kernel divides in float32 (as quant_i8.cu
does). So scales are compared to f32 precision and a code may differ by 1 only
where w/scale sits on a .5 tie in one precision but not the other.
"""
from __future__ import annotations
import random
import struct
import pytest
from dtype_io import pack_bf16, unpack_bf16
from nf4 import dequant_int8, i8_signed, max_abs_err, pack_f32, quantize_int8_symmetric, unpack_f32
import mojo_quant

pytestmark = pytest.mark.skipif(
    not mojo_quant.mojo_quant_available(), reason="mojo/quant_i8_mojo.so not built"
)
B = 64


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _weights(n: int, seed: int) -> list:
    rng = random.Random(seed)
    w = [rng.gauss(0.0, 0.02) for _ in range(n)]
    # Outliers and an all-zero block, the cases that move the scale.
    if n > 3 * B:
        w[5] = 0.9
        w[B + 7] = -1.7
        for i in range(2 * B, 3 * B):
            w[i] = 0.0
    return w


def _check(vals: list, got) -> tuple:
    assert got is not None
    q, sc = got
    gq, gsc = quantize_int8_symmetric(vals, blocksize=B)
    assert len(q) == len(gq) and len(sc) == len(gsc)
    for b, (s, g) in enumerate(zip(sc, gsc)):
        assert s == pytest.approx(g, rel=2e-7, abs=0.0), b
    off_by_one = 0
    for i, (a, g) in enumerate(zip(q, gq)):
        a, g = i8_signed(a), i8_signed(g)
        assert -127 <= a <= 127
        if a == g:
            continue
        frac = abs(vals[i] / gsc[i // B]) % 1.0
        assert abs(a - g) == 1 and abs(frac - 0.5) < 1e-4, (i, a, g, vals[i])
        off_by_one += 1
    rec = dequant_int8(q, sc, B)
    err = max_abs_err(vals, rec)
    # Never worse than half a step of its block (+ f32 slack).
    for i, (v, r) in enumerate(zip(vals, rec)):
        assert abs(v - r) <= 0.5 * sc[i // B] * (1 + 1e-6) + 1e-12, i
    return err, off_by_one


@pytest.mark.parametrize("n", [1, 63, 64, 65, 4096 + 17])
def test_mojo_bf16_cpu_golden(n):
    raw = pack_bf16(_weights(n, seed=n))
    vals = unpack_bf16(raw)
    _, off = _check(vals, mojo_quant.quant_bf16_i8(raw, device="cpu"))
    assert off <= max(1, n // 1000)


@pytest.mark.parametrize("n", [1, 63, 64, 65, 4096 + 17])
def test_mojo_f32_cpu_golden(n):
    raw = pack_f32(_weights(n, seed=1000 + n))
    vals = unpack_f32(raw)
    _, off = _check(vals, mojo_quant.quant_f32_i8(raw, device="cpu"))
    assert off <= max(1, n // 1000)


def test_mojo_cpu_zero_block_and_clip():
    vals = [0.0] * B + [1.0, -1.0] + [0.25] * (B - 2)
    q, sc = mojo_quant.quant_f32_i8(pack_f32(vals), device="cpu")
    assert sc[0] == 1.0
    assert all(b == 0 for b in q[:B])
    assert sc[1] == pytest.approx(1.0 / 127.0, rel=1e-7)
    assert i8_signed(q[B]) == 127 and i8_signed(q[B + 1]) == -127
    assert i8_signed(q[B + 2]) == 32  # 0.25 * 127 = 31.75


def test_mojo_cpu_round_half_even():
    # amax 127 -> scale exactly 1.0, so w is the pre-round value.
    vals = [127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.5]
    q, sc = mojo_quant.quant_f32_i8(pack_f32(vals), device="cpu")
    assert sc[0] == 1.0
    assert [i8_signed(b) for b in q] == [127, 0, 2, 2, 0, -2, -2, 4]


def test_mojo_cpu_rejects_bad_args():
    assert mojo_quant.quant_f32_i8(b"", device="cpu") is None
    assert mojo_quant.quant_f32_i8(pack_f32([1.0]), blocksize=32, device="cpu") is None
    assert mojo_quant.quant_bf16_i8(b"\x00\x00\x00", device="cpu") is None
    assert mojo_quant.quant_f32_i8(pack_f32([1.0]), device="cuda") is None


def test_mojo_gpu_without_device_returns_none_or_quantizes():
    # rc 6 (no device) maps to None; on a GPU host it must match the CPU path.
    raw = pack_f32(_weights(4096 + 17, seed=7))
    got = mojo_quant.quant_f32_i8(raw, device="gpu")
    if got is not None:
        assert got == mojo_quant.quant_f32_i8(raw, device="cpu")
