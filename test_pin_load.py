#!/usr/bin/env python3
"""pin_load: fail-closed ABI check, host dequant, CPU compare vs source."""
from __future__ import annotations
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
import fast
from dtype_io import pack_bf16, unpack_bf16
from nf4 import pack_f32, quantize_int8_symmetric, dequant_int8
from pin_convert import Policy, convert_pin
from pin_load import Pin, verify_pin
from safetensors_io import SafeTensorsFile, write_safetensors

ROOT = Path(__file__).resolve().parent
OUT_F, IN_F = 8, 96


def _src(td: Path) -> tuple:
    rng = random.Random(1)
    w = unpack_bf16(pack_bf16([rng.gauss(0.0, 0.05) for _ in range(OUT_F * IN_F)]))
    src = td / "src"
    src.mkdir()
    write_safetensors(
        str(src / "model.safetensors"),
        [
            ("model.layers.0.mlp.down_proj.weight", "BF16", (OUT_F, IN_F), pack_bf16(w)),
            ("model.embed_tokens.weight", "BF16", (4, 8), pack_bf16([0.25] * 32)),
            ("model.norm.weight", "BF16", (8,), b"\x80\x3f" * 8),
        ],
    )
    (src / "config.json").write_text(json.dumps({"model_type": "phi3"}) + "\n")
    return src, w


def _rewrite(pin_dir: Path, mutate) -> None:
    """Rewrite model.safetensors with mutate(list_of_tensors)."""
    p = pin_dir / "model.safetensors"
    with SafeTensorsFile(str(p)) as st:
        items = [(n, i.dtype, i.shape, st.read_bytes(n)) for n, i in st.tensors.items()]
    write_safetensors(str(p), mutate(items))


def test_verify_all_dests_and_dequant_matches_reference():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, w = _src(td)
        for dest in ("int8", "nf4", "nested-nf8"):
            out = td / dest
            convert_pin(src, out, policy=Policy(dest=dest))
            rep = verify_pin(out, expand=True)
            assert rep["ok"], rep["errors"]
            assert rep["n_linears"] == 1
            assert rep["n_passthrough"] == 2  # embed + norm stay BF16
            assert rep["expanded"]["n_elem"] == OUT_F * IN_F
            with Pin(out) as pin:
                got = [float(x) for x in pin.dequant("model.layers.0.mlp.down_proj")]
            if dest == "int8":
                q, s = quantize_int8_symmetric(w, 64)
                ref = dequant_int8(q, _f32(s), 64)  # scales are stored as F32
                assert got == ref
            err = max(abs(a - b) for a, b in zip(got, w))
            assert err < (0.02 if dest == "nf4" else 0.002)


def _f32(vals):
    from nf4 import unpack_f32
    return unpack_f32(pack_f32(vals))


def test_verify_refuses_bad_pins():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, _ = _src(td)
        out = td / "pin"
        convert_pin(src, out, policy=Policy(dest="int8"))
        stem = "model.layers.0.mlp.down_proj"

        # unknown schema
        pj = json.loads((out / "pin.json").read_text())
        (out / "pin.json").write_text(json.dumps({**pj, "schema": "int8_pin_v9"}))
        rep = verify_pin(out)
        assert not rep["ok"] and "schema" in rep["errors"][0]
        (out / "pin.json").write_text(json.dumps(pj))
        assert verify_pin(out)["ok"]

        # scale count wrong
        _rewrite(out, lambda it: [
            (n, d, (sh[0] - 1,), b[:-4]) if n == stem + ".weight.int8_scale" else (n, d, sh, b)
            for n, d, sh, b in it
        ])
        rep = verify_pin(out)
        assert not rep["ok"]
        assert any("int8_scale" in e for e in rep["errors"])

        # missing scale entirely + orphan state check
        convert_pin(src, out, policy=Policy(dest="int8"))
        _rewrite(out, lambda it: [t for t in it if t[0] != stem + ".weight.int8_scale"])
        rep = verify_pin(out)
        assert not rep["ok"] and any("missing" in e for e in rep["errors"])

        # state JSON garbage
        convert_pin(src, out, policy=Policy(dest="int8"))
        _rewrite(out, lambda it: [
            (n, d, (3,), b"{x}") if n == stem + ".weight.int8_state" else (n, d, sh, b)
            for n, d, sh, b in it
        ])
        rep = verify_pin(out)
        assert not rep["ok"] and any("JSON" in e for e in rep["errors"])

        # zero scale
        convert_pin(src, out, policy=Policy(dest="int8"))
        _rewrite(out, lambda it: [
            (n, d, sh, b"\x00" * len(b)) if n == stem + ".weight.int8_scale" else (n, d, sh, b)
            for n, d, sh, b in it
        ])
        rep = verify_pin(out)
        assert not rep["ok"] and any("non-positive" in e for e in rep["errors"])

        # missing model file
        (out / "model.safetensors").unlink()
        assert not verify_pin(out)["ok"]


def test_verify_refuses_hand_edited_nf8_cells():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, _ = _src(td)
        out = td / "nest"
        convert_pin(src, out, policy=Policy(dest="nested-nf8"))
        _rewrite(out, lambda it: [
            (n, d, sh, b"\x00\x00\x80\x3f" + b[4:]) if n == "_nf8_cells" else (n, d, sh, b)
            for n, d, sh, b in it
        ])
        rep = verify_pin(out)
        assert not rep["ok"] and any("_nf8_cells" in e for e in rep["errors"])


def test_verify_cli_exit_codes():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, _ = _src(td)
        out = td / "pin"
        convert_pin(src, out, policy=Policy(dest="int8"))
        ok = subprocess.run([sys.executable, str(ROOT / "bf16_to_int8.py"), "verify", "--pin", str(out)],
                            capture_output=True, text=True)
        assert ok.returncode == 0 and json.loads(ok.stdout)["ok"]
        bad = subprocess.run([sys.executable, str(ROOT / "bf16_to_int8.py"), "verify", "--pin", str(td)],
                             capture_output=True, text=True)
        assert bad.returncode == 3


def test_config_quant_method_follows_dest():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, _ = _src(td)
        want = {"int8": "bf16_to_int8_pin", "nf4": "bf16_to_nf4_pin", "nested-nf8": "nested_nf8_pin"}
        for dest, method in want.items():
            out = td / dest
            convert_pin(src, out, policy=Policy(dest=dest))
            q = json.loads((out / "config.json").read_text())["quantization_config"]
            assert q["quant_method"] == method
            assert q["schema"] == json.loads((out / "pin.json").read_text())["schema"]
            assert ("load_in_8bit" in q) == (dest == "int8")


def test_compare_cpu_transfer_exact():
    import compare_pin_vs_src
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, _ = _src(td)
        out = td / "pin"
        convert_pin(src, out, policy=Policy(dest="int8"))
        rc = compare_pin_vs_src.main(["--pin", str(out), "--device", "cpu"])
        rep = json.loads((out / "COMPARE_VS_BF16.json").read_text())
        assert rc == 0
        assert rep["transfer_exact"] and rep["scheme_within_half_lsb"]
        assert rep["device"] == "cpu"
        assert rep["n_linears"] == 1 and rep["rmse_vs_bf16"] > 0.0
        # flip one I8 byte → transfer no longer exact
        stem = "model.layers.0.mlp.down_proj.weight"
        _rewrite(out, lambda it: [
            (n, d, sh, bytes([(b[0] + 1) & 0xFF]) + b[1:]) if n == stem else (n, d, sh, b)
            for n, d, sh, b in it
        ])
        rc = compare_pin_vs_src.main(["--pin", str(out), "--device", "cpu"])
        rep = json.loads((out / "COMPARE_VS_BF16.json").read_text())
        assert rc == 3 and rep["i8_byte_mismatch_total"] == 1


def test_cpu_compare_twins_agree():
    """numpy and stdlib compare give the same counts."""
    import compare_pin_vs_src as c
    rng = random.Random(3)
    raw = pack_bf16([rng.gauss(0, 1) for _ in range(200)])
    q, s = c.cpu_quant_bf16_i8(raw)
    a = c.cpu_compare_bf16_i8(raw, q, s)
    assert a["n_over_half_scale"] == 0
    if fast.available():
        np_saved = fast.np
        try:
            fast.np = None
            q2, s2 = c.cpu_quant_bf16_i8(raw)
            b = c.cpu_compare_bf16_i8(raw, q2, s2)
        finally:
            fast.np = np_saved
        assert q2 == q and s2 == s
        assert b["n_over_half_scale"] == a["n_over_half_scale"]
        assert abs(b["max_abs"] - a["max_abs"]) < 1e-12
        assert abs(b["sum_sq"] - a["sum_sq"]) < 1e-9
