"""Mojo INT8 block-quant (twin of gpu_quant.py). None if the module is not built.

Build with `bash mojo/build.sh` (needs `pip install mojo max`). Same contract as
gpu_quant.quant_bf16_i8: (q8 bytes, per-block scales) or None.
"""
from __future__ import annotations
import ctypes
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_MOD = None
_MOD_TRIED = False
_DIR = Path(__file__).resolve().parent / "mojo"


def mojo_quant_available() -> bool:
    global _MOD, _MOD_TRIED
    if _MOD is not None:
        return True
    if _MOD_TRIED:
        return False
    _MOD_TRIED = True
    if not (_DIR / "quant_i8_mojo.so").is_file():
        return False
    sys.path.insert(0, str(_DIR))
    try:
        import quant_i8_mojo  # type: ignore[import-not-found]
    except ImportError:
        return False
    finally:
        sys.path.remove(str(_DIR))
    _MOD = quant_i8_mojo
    return True


def _quant(raw: bytes, elem_size: int, fn_name: str, blocksize: int) -> Optional[Tuple[bytes, List[float]]]:
    if blocksize != 64 or not mojo_quant_available() or _MOD is None:
        return None
    n = len(raw) // elem_size
    if n < 1 or len(raw) != n * elem_size:
        return None
    nblk = (n + blocksize - 1) // blocksize
    buf = ctypes.create_string_buffer(raw, len(raw))
    q = (ctypes.c_int8 * n)()
    sc = (ctypes.c_float * nblk)()
    rc = getattr(_MOD, fn_name)(ctypes.addressof(buf), n, ctypes.addressof(q), ctypes.addressof(sc))
    if rc != 0:
        return None
    return bytes(q), [float(sc[i]) for i in range(nblk)]


def quant_bf16_i8(raw: bytes, blocksize: int = 64, device: str = "cpu") -> Optional[Tuple[bytes, List[float]]]:
    """BF16 little-endian bytes → (int8 bytes, scales). device: cpu | gpu."""
    return _quant(raw, 2, f"quant_bf16_i8_{device}", blocksize)


def quant_f32_i8(raw: bytes, blocksize: int = 64, device: str = "cpu") -> Optional[Tuple[bytes, List[float]]]:
    """F32 little-endian bytes → (int8 bytes, scales). device: cpu | gpu."""
    return _quant(raw, 4, f"quant_f32_i8_{device}", blocksize)
