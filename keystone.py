"""Keystone hole+plug: split BF16 bitplanes. Inflate is bitwise, not a codebook.
VRAM hole = 4 MSBs/weight packed. RAM plug = 12 LSBs.
Complete plug → bit-exact BF16.
"""
from __future__ import annotations
import struct
from typing import List, Sequence, Tuple
from dtype_io import pack_bf16, unpack_bf16
def split_bf16(vals: Sequence[float]) -> Tuple[bytes, bytes]:
    raw = pack_bf16(vals)
    n = len(vals)
    hole = bytearray((n + 1) // 2) # 4 bits each
    plug = bytearray(n * 2) # 12 bits in u16 low
    for i in range(n):
        u16 = struct.unpack_from("<H", raw, i * 2)[0]
        hi4 = (u16 >> 12) & 0xF
        lo12 = u16 & 0x0FFF
        if i & 1 == 0:
            hole[i // 2] = (hole[i // 2] & 0xF0) | hi4
        else:
            hole[i // 2] = (hole[i // 2] & 0x0F) | (hi4 << 4)
        struct.pack_into("<H", plug, i * 2, lo12)
    return bytes(hole), bytes(plug)
def inflate(hole: bytes, plug: bytes, n: int) -> List[float]:
    raw = bytearray(n * 2)
    for i in range(n):
        b = hole[i // 2]
        hi4 = (b & 0x0F) if (i & 1) == 0 else ((b >> 4) & 0x0F)
        lo12 = struct.unpack_from("<H", plug, i * 2)[0] & 0x0FFF
        struct.pack_into("<H", raw, i * 2, (hi4 << 12) | lo12)
    return unpack_bf16(bytes(raw))
def hole_only(hole: bytes, n: int) -> List[float]:
    """Missing plug: LSBs zero. Legal BF16, wrong values. MMA can still run."""
    raw = bytearray(n * 2)
    for i in range(n):
        b = hole[i // 2]
        hi4 = (b & 0x0F) if (i & 1) == 0 else ((b >> 4) & 0x0F)
        struct.pack_into("<H", raw, i * 2, hi4 << 12)
    return unpack_bf16(bytes(raw))
if __name__ == "__main__":
    from nf4_to_int8 import demo_weights
    w = demo_weights(256)
    hole, plug = split_bf16(w)
    rec = inflate(hole, plug, len(w))
    packed = pack_bf16(w)
    rec_p = pack_bf16(rec)
    exact = packed == rec_p
    blur = hole_only(hole, len(w))
    print(
        {
            "n": len(w),
            "vram_hole_bytes": len(hole),
            "ram_plug_bytes": len(plug),
            "bits_vram": 8 * len(hole) / len(w),
            "inflate_bit_exact_bf16": exact,
            "hole_only_not_exact": pack_bf16(blur) != packed,
        }
    )
