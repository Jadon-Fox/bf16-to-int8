#!/usr/bin/env python3
"""numpy fast path must write the same bytes as the stdlib loops."""
from __future__ import annotations
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest
import fast
from dtype_io import pack_bf16, unpack_bf16
from nested_nf import NF8_CELLS, encode_nested
from nf4 import NF4_CODEBOOK, dequant_nf4, pack_f32, pack_indices, quantize_int8_symmetric, quantize_nf4
from safetensors_io import SafeTensorsFile, write_safetensors

ROOT = Path(__file__).resolve().parent
needs_np = pytest.mark.skipif(not fast.available(), reason="numpy not installed")


def _weights(n: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    w = [rng.gauss(0.0, 0.02) for _ in range(n)]
    w[64:128] = [0.0] * 64  # all-zero block → scale 1.0 / idx 7
    w[200] = 0.5  # outlier block
    w[300:304] = [NF4_CODEBOOK[3] * 0.1, -0.1, 0.1, 0.05]
    return unpack_bf16(pack_bf16(w))  # exact BF16 values


@needs_np
@pytest.mark.parametrize("n", [1, 63, 64, 65, 1000, 4097])
def test_int8_parity(n):
    w = _weights(max(n, 400))[:n]
    q_ref, s_ref = quantize_int8_symmetric(w, 64)
    q, s = fast.quantize_int8(fast.np.asarray(w, dtype=fast.np.float64), 64)
    assert q.tobytes() == bytes(q_ref)
    assert s.tolist() == s_ref
    assert fast.pack_f32(s) == pack_f32(s_ref)


@needs_np
@pytest.mark.parametrize("n", [1, 64, 65, 1000, 4097])
def test_nf4_parity(n):
    w = _weights(max(n, 400))[:n]
    p_ref, a_ref = quantize_nf4(w, 64)
    p, a = fast.quantize_nf4(fast.np.asarray(w, dtype=fast.np.float64), 64)
    assert p == bytes(p_ref)
    assert a.tolist() == a_ref
    assert fast.dequant_nf4(p, a, n, 64).tolist() == dequant_nf4(p_ref, a_ref, n, 64)


@needs_np
@pytest.mark.parametrize("n", [1, 65, 1000])
def test_nested_parity(n):
    w = _weights(max(n, 400))[:n]
    i_ref, p_ref, a_ref = encode_nested(w, 64)
    i, p, a = fast.encode_nested(fast.np.asarray(w, dtype=fast.np.float64), NF8_CELLS, 64)
    assert fast.pack_nibbles(i) == bytes(pack_indices(i_ref))
    assert fast.pack_nibbles(p) == bytes(pack_indices(p_ref))
    assert a.tolist() == a_ref


@needs_np
def test_unpack_f16_subnormal_inf():
    from dtype_io import unpack_f16
    raw = bytes([0x01, 0x00, 0x00, 0x7C, 0x00, 0xFC, 0xFF, 0x3B])
    assert fast.unpack_dense("F16", raw, 4).tolist() == unpack_f16(raw)


def _write_src(src: Path, w: list) -> None:
    out_f, in_f = 16, len(w) // 16
    write_safetensors(
        str(src / "model.safetensors"),
        [
            ("model.layers.0.mlp.down_proj.weight", "BF16", (out_f, in_f), pack_bf16(w)),
            ("model.layers.0.self_attn.o_proj.weight", "F32", (out_f, in_f), pack_f32(w)),
            ("model.norm.weight", "BF16", (3,), b"\x80\x3f" * 3),
        ],
        metadata={"format": "pt"},
    )
    (src / "config.json").write_text(json.dumps({"model_type": "phi3"}) + "\n")


@needs_np
@pytest.mark.parametrize("dest", ["int8", "nf4", "nested-nf8"])
def test_pin_bytes_same_with_and_without_numpy(dest):
    """Whole-pin parity: run the CLI in a subprocess with numpy on, then off."""
    w = _weights(16 * 72)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        _write_src(src, w)
        blobs = {}
        for flag in ("", "1"):
            out = Path(td) / f"pin{flag or 'np'}"
            env = {"BF16_TO_INT8_NO_NUMPY": flag} if flag else {}
            subprocess.run(
                [sys.executable, str(ROOT / "bf16_to_int8.py"), "pin", "--src", str(src),
                 "--out", str(out), "--to", dest],
                check=True, capture_output=True, text=True,
                env={**{k: v for k, v in os.environ.items() if k != "BF16_TO_INT8_NO_NUMPY"}, **env},
            )
            blobs[flag] = (out / "model.safetensors").read_bytes()
        assert blobs[""] == blobs["1"]


def test_safetensors_no_holes_and_aligned():
    """Spec: contiguous data_offsets. HF safetensors refuses gaps."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.safetensors"
        write_safetensors(
            str(p),
            [
                ("a", "U8", (3,), b"abc"),
                ("b", "F32", (2,), pack_f32([1.0, 2.0])),
                ("c", "BF16", (1,), b"\x80\x3f"),
                ("d", "I8", (5,), b"\x01\x02\x03\x04\x05"),
            ],
        )
        with SafeTensorsFile(str(p)) as st:
            spans = sorted(i.data_offsets for i in st.tensors.values())
            assert spans[0][0] == 0
            for (_, e), (s, _) in zip(spans, spans[1:]):
                assert s == e
            assert st.data_start % 8 == 0
            assert p.stat().st_size == st.data_start + spans[-1][1]
            for info in st.tensors.values():
                width = {"F32": 4, "BF16": 2}.get(info.dtype, 1)
                assert info.data_offsets[0] % width == 0
            assert st.read_bytes("a") == b"abc"
            assert st.read_bytes("d") == b"\x01\x02\x03\x04\x05"


def test_safetensors_rejects_size_mismatch():
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError):
            write_safetensors(str(Path(td) / "x.safetensors"), [("a", "F32", (2,), b"\x00" * 4)])


def test_hf_safetensors_reads_pin():
    hf = pytest.importorskip("safetensors")
    w = _weights(16 * 72)
    from pin_convert import Policy, convert_pin
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        _write_src(src, w)
        for dest in ("int8", "nf4", "nested-nf8"):
            out = Path(td) / dest
            convert_pin(src, out, policy=Policy(dest=dest))
            # safe_open validates the header (offsets, no holes). numpy has no bf16,
            # so read back only the non-BF16 tensors.
            with hf.safe_open(str(out / "model.safetensors"), framework="np") as f:
                keys = set(f.keys())
                assert "model.norm.weight" in keys
                with SafeTensorsFile(str(out / "model.safetensors")) as ours:
                    for k in keys:
                        if ours.tensors[k].dtype != "BF16":
                            assert f.get_tensor(k).tobytes() == ours.read_bytes(k)
