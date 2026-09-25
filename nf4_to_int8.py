#!/usr/bin/env python3
"""Hangover filename. Documented alias of bf16_to_int8.py. Dest hop is BF16/F16 → INT8.
  python3 bf16_to_int8.py pin --src /path/to/phi-4-mini-bf16 --out /path/to/int8-pin
  python3 nf4_to_int8.py pin --src /path/to/phi-4-mini-bf16 --out /path/to/int8-pin
NF4/bnb-4bit sources need --allow-requant (lossy). Product hop is dense BF16.
Offline pin. Not a dest-pack flip. Not .
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from nf4 import (
    NIBBLE_HI_THEN_LO,
    NIBBLE_LO_THEN_HI,
    dequant_int8,
    dequant_nf4,
    max_abs_err,
    n_absmax,
    pack_f32,
    quantize_int8_symmetric,
    quantize_nf4,
    qweight_nbytes,
    rmse,
    unpack_f32,
)
def write_outputs(prefix: Path, q: bytes, scales: list[float], meta: dict) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_suffix(".i8.bin")).write_bytes(bytes(q))
    (prefix.with_suffix(".scale.f32.bin")).write_bytes(pack_f32(scales))
    (prefix.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2) + "\n")
def convert(
    qweight: bytes,
    absmax: list[float],
    n_elem: int,
    blocksize: int,
    nibble_order: int,
    int8_blocksize: int,
) -> dict:
    f32 = dequant_nf4(qweight, absmax, n_elem, blocksize, nibble_order)
    q8, scales = quantize_int8_symmetric(f32, blocksize=int8_blocksize)
    recon = dequant_int8(q8, scales, blocksize=int8_blocksize)
    meta = {
        "schema": "nf4_to_int8_v1",
        "n_elem": n_elem,
        "nf4_blocksize": blocksize,
        "int8_blocksize": int8_blocksize,
        "nibble_order": "lo_then_hi" if nibble_order == NIBBLE_LO_THEN_HI else "hi_then_lo",
        "int8_scheme": "symmetric_per_block_zp0",
        "qmin": -127,
        "qmax": 127,
        "n_int8_scales": len(scales),
        "rmse_vs_nf4_dequant": rmse(f32, recon),
        "max_abs_err_vs_nf4_dequant": max_abs_err(f32, recon),
        "note": "requant of NF4 dequant. not a claim. not ORCH_BASE_PACK.",
    }
    return {"q8": q8, "scales": scales, "f32": f32, "recon": recon, "meta": meta}
def demo_weights(n: int = 256) -> list[float]:
    out = []
    for i in range(n):
        x = ((i * 37) % 200 - 100) / 80.0
        out.append(x * (0.4 + (i % 17) / 40.0))
    return out
def run_tensor(args: argparse.Namespace) -> int:
    nibble = NIBBLE_LO_THEN_HI if args.nibble_order == "lo_then_hi" else NIBBLE_HI_THEN_LO
    i8_bs = args.int8_blocksize if args.int8_blocksize > 0 else args.blocksize
    if args.demo:
        w = demo_weights(256)
        qw, am = quantize_nf4(w, blocksize=args.blocksize, nibble_order=nibble)
        n_elem = len(w)
        qweight = bytes(qw)
        absmax = am
    else:
        if not args.qweight or not args.absmax or not args.n_elem:
            print("HARD_BLOCK: need --qweight --absmax --n-elem, or --demo", file=sys.stderr)
            return 2
        qweight = args.qweight.read_bytes()
        absmax = unpack_f32(args.absmax.read_bytes())
        n_elem = args.n_elem
        if len(qweight) < qweight_nbytes(n_elem):
            print("HARD_BLOCK: qweight shorter than n_elem/2", file=sys.stderr)
            return 2
        if len(absmax) < n_absmax(n_elem, args.blocksize):
            print("HARD_BLOCK: absmax shorter than n_blocks", file=sys.stderr)
            return 2
    got = convert(qweight, absmax, n_elem, args.blocksize, nibble, i8_bs)
    write_outputs(args.out_prefix, got["q8"], got["scales"], got["meta"])
    if args.json:
        print(json.dumps(got["meta"], indent=2))
    else:
        m = got["meta"]
        print(
            f"nf4→int8 n={m['n_elem']} nf4_bs={m['nf4_blocksize']} "
            f"i8_bs={m['int8_blocksize']} rmse={m['rmse_vs_nf4_dequant']:.6g} "
            f"max|e|={m['max_abs_err_vs_nf4_dequant']:.6g}"
        )
        print(f"wrote {args.out_prefix}.i8.bin {args.out_prefix}.scale.f32.bin {args.out_prefix}.meta.json")
        print("requant only.")
    return 0
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="BF16/F16 → INT8 or NF4 pin (one hop)")
    sub = p.add_subparsers(dest="cmd")
    pin_p = sub.add_parser("pin", help="one-hop convert: dense BF16/F16/F32 → INT8 or NF4 pin")
    pin_p.add_argument("--src", required=True, help="HF model dir or model.safetensors (BF16/F16)")
    pin_p.add_argument("--out", required=True, help="output pin directory")
    pin_p.add_argument("--to", choices=("int8", "nf4", "nested-nf8"), default="int8", help="dest pin. default int8")
    pin_p.add_argument("--int8-blocksize", type=int, default=64, help="block size for INT8 or NF4")
    pin_p.add_argument("--dry-run", action="store_true")
    pin_p.add_argument(
        "--allow-requant",
        action="store_true",
        help="permit NF4/FP4 sources (lossy second hop). off by default",
    )
    pin_p.add_argument(
        "--dense",
        choices=("quantize", "copy", "int8"),
        default="quantize",
        help="16-bit linears. default quantize to --to",
    )
    pin_p.add_argument(
        "--embed",
        choices=("copy", "quantize", "int8"),
        default="copy",
        help="token embeddings / lm_head. default copy",
    )
    fix_p = sub.add_parser("fixture", help="write a tiny 8x64 INT8 pin for orch loader CI")
    fix_p.add_argument("--out", required=True, help="directory; writes _src/ and int8_pin/")
    ten_p = sub.add_parser("tensor", help="convert one packed NF4 buffer")
    ten_p.add_argument("--qweight", type=Path)
    ten_p.add_argument("--absmax", type=Path)
    ten_p.add_argument("--n-elem", type=int)
    ten_p.add_argument("--blocksize", type=int, default=64)
    ten_p.add_argument("--int8-blocksize", type=int, default=0)
    ten_p.add_argument("--nibble-order", choices=("lo_then_hi", "hi_then_lo"), default="lo_then_hi")
    ten_p.add_argument("--out-prefix", type=Path, default=Path("out/converted"))
    ten_p.add_argument("--demo", action="store_true")
    ten_p.add_argument("--json", action="store_true")
    # legacy flags (no subcommand)
    p.add_argument("--qweight", type=Path)
    p.add_argument("--absmax", type=Path)
    p.add_argument("--n-elem", type=int)
    p.add_argument("--blocksize", type=int, default=64)
    p.add_argument("--int8-blocksize", type=int, default=0)
    p.add_argument("--nibble-order", choices=("lo_then_hi", "hi_then_lo"), default="lo_then_hi")
    p.add_argument("--out-prefix", type=Path, default=Path("out/converted"))
    p.add_argument("--demo", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "pin":
        from pin_convert import cmd_pin
        return cmd_pin(args)
    if args.cmd == "fixture":
        from pin_convert import write_tiny_fixture
        got = write_tiny_fixture(Path(args.out))
        print(json.dumps(got["pin"], indent=2))
        return 0
    if args.cmd == "tensor":
        return run_tensor(args)
    return run_tensor(args)
if __name__ == "__main__":
    raise SystemExit(main())
