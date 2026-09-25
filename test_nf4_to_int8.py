#!/usr/bin/env python3
"""Hangover filename. Golden checks for bf16_to_int8 / nf4_to_int8 alias. Not ."""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from nf4 import (
    NF4_CODEBOOK,
    NIBBLE_LO_THEN_HI,
    decode_absmax_double,
    dequant_int8,
    dequant_nf4,
    max_abs_err,
    pack_f32,
    quantize_int8_symmetric,
    quantize_nf4,
    rmse,
    unpack_f32,
    unpack_indices,
)
from nf4_to_int8 import convert, demo_weights
from dtype_io import pack_bf16, unpack_bf16
from pin_convert import Policy, convert_pin
from safetensors_io import SafeTensorsFile, write_safetensors
ROOT = Path(__file__).resolve().parent

def assert_no_empty_key(obj, where="root"):
    """Sep 11 sweep damage guard (issue #10): emitted artifacts must not carry "" keys."""
    if isinstance(obj, dict):
        assert "" not in obj, f"empty key at {where}"
        for k, v in obj.items():
            assert_no_empty_key(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_no_empty_key(v, f"{where}[{i}]")


def test_codebook_ends():
    assert NF4_CODEBOOK[0] == -1.0
    assert NF4_CODEBOOK[7] == 0.0
    assert NF4_CODEBOOK[15] == 1.0
    assert len(NF4_CODEBOOK) == 16
def test_nf4_roundtrip_zero_and_scale():
    w = [0.0] * 64 + [0.5] * 64
    qw, am = quantize_nf4(w, blocksize=64)
    rec = dequant_nf4(qw, am, 128, 64, NIBBLE_LO_THEN_HI)
    assert am[0] == 0.0
    assert abs(am[1] - 0.5) < 1e-6
    assert max(abs(x) for x in rec[:64]) < 1e-12
    assert abs(rec[64] - 0.5) < 1e-6
def test_int8_zero_point_and_clip():
    w = [0.0, 1.0, -1.0, 2.0]
    q, s = quantize_int8_symmetric(w, blocksize=4)
    rec = dequant_int8(q, s, 4)
    assert s[0] == 2.0 / 127.0
    signed = [b if b < 128 else b - 256 for b in q]
    assert signed[0] == 0
    assert signed[3] == 127
    assert abs(rec[3] - 2.0) < 1e-6
def test_convert_rmse_small():
    w = demo_weights(256)
    qw, am = quantize_nf4(w, blocksize=64)
    got = convert(bytes(qw), am, 256, 64, NIBBLE_LO_THEN_HI, 64)
    assert_no_empty_key(got["meta"], "meta")
    assert got["meta"]["rmse_vs_nf4_dequant"] < 0.02
    assert got["meta"]["max_abs_err_vs_nf4_dequant"] < 0.05
    assert rmse(got["f32"], got["recon"]) == got["meta"]["rmse_vs_nf4_dequant"]
    assert max_abs_err(got["f32"], got["recon"]) == got["meta"]["max_abs_err_vs_nf4_dequant"]
def test_cli_demo():
    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td) / "w"
        r = subprocess.run(
            [sys.executable, str(ROOT / "nf4_to_int8.py"), "--demo", "--out-prefix", str(prefix), "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        meta = json.loads(r.stdout)
        assert meta["n_elem"] == 256
        assert (prefix.with_suffix(".i8.bin")).stat().st_size == 256
        assert (prefix.with_suffix(".scale.f32.bin")).stat().st_size == 4 * 4
def test_double_quant_offset():
    # 4 L1 blocks, nested_bs=4 so 1 nested scale
    nmap = [((i - 127) / 127.0) for i in range(256)]
    nmap[127] = 0.0
    codes = bytes([200, 180, 127, 90])
    nested = [0.04]
    off = 0.0855
    am = decode_absmax_double(codes, nmap, nested, off, nested_blocksize=4)
    assert len(am) == 4
    assert am[2] == off # code 127 → 0 * scale + offset
    assert am[0] > am[3]
def _write_single_quant_module(path: Path, stem: str, w: list[float], out_f: int, in_f: int) -> None:
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
    tensors = [
        (stem + ".weight", "U8", (len(qw), 1), bytes(qw)),
        (stem + ".weight.absmax", "F32", (len(am),), pack_f32(am)),
        (stem + ".weight.quant_map", "F32", (16,), pack_f32(list(NF4_CODEBOOK))),
        (stem + ".weight.quant_state.bitsandbytes__nf4", "U8", (len(state),), state),
    ]
    # BF16-looking passthrough (raw 2 bytes × 4)
    tensors.append(("model.norm.weight", "BF16", (4,), b"\x00\x3c" * 4))
    write_safetensors(str(path), tensors, metadata={"format": "pt"})
def test_pin_single_quant_roundtrip():
    out_f, in_f = 8, 64
    w = demo_weights(out_f * in_f)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        _write_single_quant_module(src / "model.safetensors", "model.layers.0.mlp.down_proj", w, out_f, in_f)
        (src / "config.json").write_text(json.dumps({"model_type": "phi3", "hidden_size": 8}) + "\n")
        dst = Path(td) / "pin"
        got = convert_pin(src, dst, int8_blocksize=64, policy=Policy(allow_requant=True))
        pin = got["pin"]
        assert pin["n_nf4_modules"] == 1
        assert pin["n_passthrough"] == 1
        assert pin["rmse_mean"] < 0.02
        assert (dst / "model.safetensors").is_file()
        assert (dst / "pin.json").is_file()
        assert_no_empty_key(json.loads((dst / "pin.json").read_text()), "pin.json")
        with SafeTensorsFile(str(dst / "model.safetensors")) as st:
            assert st.tensors["model.layers.0.mlp.down_proj.weight"].dtype == "I8"
            assert st.tensors["model.layers.0.mlp.down_proj.weight"].shape == (8, 64)
            assert "model.norm.weight" in st.tensors
            assert st.tensors["model.norm.weight"].dtype == "BF16"
            assert "model.layers.0.mlp.down_proj.weight.absmax" not in st.tensors
        cfg = json.loads((dst / "config.json").read_text())
        assert_no_empty_key(cfg, "config.json")
        assert cfg["quantization_config"]["quant_method"] == "bf16_to_int8_pin"
def test_cli_pin_dry_run():
    out_f, in_f = 8, 64
    w = demo_weights(out_f * in_f)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        _write_single_quant_module(src, "lin", w, out_f, in_f)
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "nf4_to_int8.py"),
                "pin",
                "--src",
                str(src),
                "--out",
                str(Path(td) / "unused"),
                "--dry-run",
                "--allow-requant",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(r.stdout)
        assert_no_empty_key(plan, "dry_run_plan")
        assert plan["n_nf4_modules"] == 1
        assert "lin" in plan["nf4_modules"]
def test_fixture_cli():
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [sys.executable, str(ROOT / "nf4_to_int8.py"), "fixture", "--out", td],
            check=True,
            capture_output=True,
            text=True,
        )
        pin = json.loads(r.stdout)
        assert_no_empty_key(pin, "fixture_pin")
        assert pin["schema"] == "bf16_to_int8_pin_v1"
        assert (Path(td) / "int8_pin" / "model.safetensors").is_file()
        assert (Path(td) / "int8_pin" / "pin.json").is_file()
def test_double_quant_module_pin():
    # encode L1 absmax through a fake 256-map so D1 decode is exact enough
    out_f, in_f = 4, 64
    w = demo_weights(out_f * in_f)
    qw, am = quantize_nf4(w, blocksize=64)
    nmap = [((i - 127) / 127.0) for i in range(256)]
    nmap[127] = 0.0
    offset = 0.08
    # pick codes + nested so decode ≈ am (am is small positive)
    # am_fp = nmap[c] * nested + offset → nmap[c] = (am - offset) / nested
    nested_val = 0.05
    codes = bytearray()
    for a in am:
        target = (a - offset) / nested_val
        best = min(range(256), key=lambda i: abs(nmap[i] - target))
        codes.append(best)
    state = json.dumps(
        {
            "quant_type": "nf4",
            "blocksize": 64,
            "dtype": "bfloat16",
            "shape": [out_f, in_f],
            "nested_blocksize": 256,
            "nested_dtype": "float32",
            "nested_offset": offset,
        },
        separators=(",", ":"),
    ).encode()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        write_safetensors(
            str(src),
            [
                ("lin.weight", "U8", (len(qw), 1), bytes(qw)),
                ("lin.weight.absmax", "U8", (len(codes),), bytes(codes)),
                ("lin.weight.nested_absmax", "F32", (1,), pack_f32([nested_val])),
                ("lin.weight.nested_quant_map", "F32", (256,), pack_f32(nmap)),
                ("lin.weight.quant_map", "F32", (16,), pack_f32(list(NF4_CODEBOOK))),
                ("lin.weight.quant_state.bitsandbytes__nf4", "U8", (len(state),), state),
            ],
        )
        dst = Path(td) / "pin"
        got = convert_pin(src, dst, policy=Policy(allow_requant=True))
        assert got["pin"]["n_nf4_modules"] == 1
        assert got["modules"][0]["double_quant"] == "double"
def test_bf16_roundtrip_bits():
    vals = [0.0, 1.0, -0.5, 0.25]
    rec = unpack_bf16(pack_bf16(vals))
    for a, b in zip(vals, rec):
        assert abs(a - b) < 1e-2
def test_dense_bf16_pin():
    out_f, in_f = 8, 64
    w = demo_weights(out_f * in_f)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        write_safetensors(
            str(src / "model.safetensors"),
            [
                ("model.layers.0.mlp.down_proj.weight", "BF16", (out_f, in_f), pack_bf16(w)),
                ("model.layers.0.mlp.gate_up_proj.weight", "BF16", (out_f, in_f), pack_bf16(w)),
                ("model.embed_tokens.weight", "BF16", (32, in_f), pack_bf16(demo_weights(32 * in_f))),
                ("model.norm.weight", "BF16", (8,), pack_bf16([1.0] * 8)),
            ],
        )
        (src / "config.json").write_text(json.dumps({"model_type": "phi3"}) + "\n")
        dst = Path(td) / "pin"
        got = convert_pin(src, dst, policy=Policy(dense="int8", embed="copy"))
        pin = got["pin"]
        assert pin["n_dense_modules"] == 2
        assert pin["n_nf4_modules"] == 0
        assert "bf16" in pin["src_kinds"]
        with SafeTensorsFile(str(dst / "model.safetensors")) as st:
            assert st.tensors["model.layers.0.mlp.down_proj.weight"].dtype == "I8"
            assert st.tensors["model.embed_tokens.weight"].dtype == "BF16"
            assert st.tensors["model.norm.weight"].dtype == "BF16"
def test_sharded_bf16_pin():
    out_f, in_f = 8, 64
    w = demo_weights(out_f * in_f)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        write_safetensors(
            str(src / "model-00001-of-00002.safetensors"),
            [("model.layers.0.mlp.down_proj.weight", "BF16", (out_f, in_f), pack_bf16(w))],
        )
        write_safetensors(
            str(src / "model-00002-of-00002.safetensors"),
            [
                ("model.layers.1.mlp.down_proj.weight", "BF16", (out_f, in_f), pack_bf16(w)),
                ("model.norm.weight", "BF16", (8,), pack_bf16([1.0] * 8)),
            ],
        )
        (src / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
                        "model.layers.1.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
                        "model.norm.weight": "model-00002-of-00002.safetensors",
                    }
                }
            )
            + "\n"
        )
        (src / "config.json").write_text(json.dumps({"model_type": "phi3"}) + "\n")
        dst = Path(td) / "pin"
        got = convert_pin(src, dst, policy=Policy(dense="int8", embed="copy"))
        pin = got["pin"]
        assert pin["n_shards"] == 2
        assert pin["n_dense_modules"] == 2
        with SafeTensorsFile(str(dst / "model.safetensors")) as st:
            assert st.tensors["model.layers.0.mlp.down_proj.weight"].dtype == "I8"
            assert st.tensors["model.layers.1.mlp.down_proj.weight"].dtype == "I8"
            assert st.tensors["model.norm.weight"].dtype == "BF16"
def test_dense_copy_policy():
    out_f, in_f = 8, 64
    w = demo_weights(out_f * in_f)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        write_safetensors(
            str(src),
            [("lin.weight", "BF16", (out_f, in_f), pack_bf16(w))],
        )
        dst = Path(td) / "pin"
        got = convert_pin(src, dst, policy=Policy(dense="copy"))
        with SafeTensorsFile(str(dst / "model.safetensors")) as st:
            assert st.tensors["lin.weight"].dtype == "BF16"
        assert got["pin"]["n_dense_modules"] == 0
def test_refuse_gptq():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        write_safetensors(
            str(src),
            [
                ("lin.qweight", "I32", (8,), b"\x00" * 32),
                ("lin.qzeros", "I32", (2,), b"\x00" * 8),
            ],
        )
        try:
            convert_pin(src, Path(td) / "pin")
        except ValueError as e:
            assert "HARD_BLOCK" in str(e)
        else:
            raise AssertionError("expected GPTQ refuse")
def test_refuse_nf4_without_allow_requant():
    out_f, in_f = 8, 64
    w = demo_weights(out_f * in_f)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        _write_single_quant_module(src, "lin", w, out_f, in_f)
        try:
            convert_pin(src, Path(td) / "pin")
        except ValueError as e:
            assert "already NF4" in str(e)
        else:
            raise AssertionError("expected NF4 refuse")
def test_bf16_to_nf4_pin():
    out_f, in_f = 8, 64
    w = demo_weights(out_f * in_f)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        write_safetensors(
            str(src / "model.safetensors"),
            [
                ("lin.weight", "BF16", (out_f, in_f), pack_bf16(w)),
                ("model.norm.weight", "BF16", (8,), pack_bf16([1.0] * 8)),
            ],
        )
        dst = Path(td) / "pin"
        got = convert_pin(src, dst, policy=Policy(dest="nf4", dense="quantize"))
        pin = got["pin"]
        assert pin["schema"] == "bf16_to_nf4_pin_v1"
        assert pin["dest"] == "nf4"
        assert pin["n_converted"] == 1
        with SafeTensorsFile(str(dst / "model.safetensors")) as st:
            assert st.tensors["lin.weight"].dtype == "U8"
            assert "lin.weight.absmax" in st.tensors
            assert st.tensors["model.norm.weight"].dtype == "BF16"
def test_nested_nf8_pin():
    from nested_nf import encode_nested
    out_f, in_f = 8, 64
    w = demo_weights(out_f * in_f)
    w_src = unpack_bf16(pack_bf16(w))
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        write_safetensors(
            str(src / "model.safetensors"),
            [
                ("lin.weight", "BF16", (out_f, in_f), pack_bf16(w)),
                ("model.norm.weight", "BF16", (8,), pack_bf16([1.0] * 8)),
            ],
        )
        dst = Path(td) / "pin"
        got = convert_pin(src, dst, policy=Policy(dest="nested-nf8", dense="quantize"))
        pin = got["pin"]
        assert pin["schema"] == "nested_nf8_pin_v1"
        assert pin["dest"] == "nested-nf8"
        assert pin["n_converted"] == 1
        with SafeTensorsFile(str(dst / "model.safetensors")) as st:
            assert "lin.weight.nf8_plug" in st.tensors
            assert "_nf8_cells" in st.tensors
            assert st.tensors["_nf8_cells"].shape == (16, 16)
            hole = unpack_indices(st.read_bytes("lin.weight"), out_f * in_f)
            plug = unpack_indices(st.read_bytes("lin.weight.nf8_plug"), out_f * in_f)
            nf4i, pl, _am = encode_nested(w_src, 64)
            assert hole == nf4i
            assert plug == pl
def test_int8_schema_strings():
    from pin_convert import INT8_PIN_SCHEMA, INT8_PIN_SCHEMA_LEGACY, _dest_schema
    assert INT8_PIN_SCHEMA == "bf16_to_int8_pin_v1"
    assert INT8_PIN_SCHEMA_LEGACY == "nf4_to_int8_pin_v1"
    assert _dest_schema("int8") == INT8_PIN_SCHEMA
    assert _dest_schema("int8") != INT8_PIN_SCHEMA_LEGACY
    assert _dest_schema("nested-nf8") == "nested_nf8_pin_v1"
def test_web_title_dest_is_bf16():
    html = (ROOT / "web" / "index.html").read_text()
    assert "<title>BF16 → INT8 pin</title>" in html
    assert "<title>NF4 → INT8 pin</title>" not in html
    assert "bf16_to_int8.py pin" in html
def test_prompt_hub_urls():
    for name in (
        "GROK_BUILD_ORCH_INT8_LOADER_PROMPT.md",
        "GROK_BUILD_ORCH_NESTED_NF8_PROMPT.md",
    ):
        text = (ROOT / name).read_text()
        assert "Jadon-Fox/nf4-to-int8" not in text
        assert "github.com/Fractal-Deployment/bf16-to-int8" in text
        assert "Jadon-Fox/training_orchestrator" not in text
        assert "Fractal-Deployment/training_orchestrator" in text
def test_gpu_compare_within_half_scale():
    from gpu_quant import compare_bf16_i8, gpu_quant_available, quant_bf16_i8
    if not gpu_quant_available():
        print("SKIP test_gpu_compare_within_half_scale (no libquant_i8.so)")
        return
    w = [0.3] * 63 + [1.0]
    raw = pack_bf16(w)
    got = quant_bf16_i8(raw, 64)
    assert got is not None
    q8, scales = got
    err = compare_bf16_i8(raw, q8, scales, 64)
    assert err is not None
    assert err["n"] == 64
    assert err["n_over_half_scale"] == 0
    assert err["rmse"] > 0.0
    half = 0.5 * scales[0]
    assert err["max_abs"] <= half + 1e-6
    zraw = pack_bf16([0.0] * 64)
    zgot = quant_bf16_i8(zraw, 64)
    assert zgot is not None
    zerr = compare_bf16_i8(zraw, zgot[0], zgot[1], 64)
    assert zerr is not None
    assert zerr["rmse"] == 0.0
    assert zerr["max_abs"] == 0.0
    assert zerr["n_over_half_scale"] == 0
if __name__ == "__main__":
    test_codebook_ends()
    test_nf4_roundtrip_zero_and_scale()
    test_int8_zero_point_and_clip()
    test_convert_rmse_small()
    test_cli_demo()
    test_double_quant_offset()
    test_pin_single_quant_roundtrip()
    test_cli_pin_dry_run()
    test_fixture_cli()
    test_double_quant_module_pin()
    test_bf16_roundtrip_bits()
    test_dense_bf16_pin()
    test_sharded_bf16_pin()
    test_dense_copy_policy()
    test_refuse_gptq()
    test_refuse_nf4_without_allow_requant()
    test_bf16_to_nf4_pin()
    test_nested_nf8_pin()
    test_int8_schema_strings()
    test_web_title_dest_is_bf16()
    test_prompt_hub_urls()
    test_gpu_compare_within_half_scale()
    print("TEST_NF4_TO_INT8_GREEN bf16_to_int8 bf16_to_nf4 nested_nf8 refuse_requant gpu_compare ")
