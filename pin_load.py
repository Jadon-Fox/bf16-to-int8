"""Reference pin reader. Fail-closed ABI check + dequant, host side.
What the orch loader (C) must accept and refuse, written down as code.
See PIN_ABI.md. Not orch hot path. Not train.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import fast
from nested_nf import NF8_CELLS, cells_flat, decode_nested
from nf4 import decode_absmax, dequant_int8, dequant_nf4, pack_f32, unpack_f32, unpack_indices
from safetensors_io import DTYPE_WIDTH, SafeTensorsFile

INT8_SCHEMAS = ("bf16_to_int8_pin_v1", "nf4_to_int8_pin_v1")
NF4_SCHEMA = "bf16_to_nf4_pin_v1"
NESTED_SCHEMA = "nested_nf8_pin_v1"
KNOWN_SCHEMAS = INT8_SCHEMAS + (NF4_SCHEMA, NESTED_SCHEMA)
INT8_AUX = (".weight.int8_scale", ".weight.int8_state")
NF4_AUX = (".weight.absmax", ".weight.quant_map", ".weight.quant_state.bitsandbytes__nf4")
# bnb double-quant extras (an NF4 source copied through with --allow-requant --to nf4)
NF4_DQ_AUX = (".weight.nested_absmax", ".weight.nested_quant_map")
NESTED_AUX = (".weight.absmax", ".weight.quant_map", ".weight.nf8_plug", ".weight.nested_state")


class PinError(ValueError):
    pass


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _json_tensor(st: SafeTensorsFile, name: str) -> dict:
    raw = st.read_bytes(name)
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PinError(f"{name}: not UTF-8 JSON ({e})") from e
    if not isinstance(obj, dict):
        raise PinError(f"{name}: JSON is not an object")
    return obj


class Pin:
    """Open pin directory. `stems` are the quantized linears (without `.weight`)."""

    def __init__(self, pin_dir: Path):
        self.dir = Path(pin_dir)
        pj = self.dir / "pin.json"
        mf = self.dir / "model.safetensors"
        if not pj.is_file():
            raise PinError(f"missing {pj}")
        if not mf.is_file():
            raise PinError(f"missing {mf}")
        try:
            self.meta = json.loads(pj.read_text())
        except json.JSONDecodeError as e:
            raise PinError(f"pin.json is not JSON ({e})") from e
        self.schema = str(self.meta.get("schema") or "")
        if self.schema not in KNOWN_SCHEMAS:
            raise PinError(f"pin.json schema={self.schema!r} not one of {list(KNOWN_SCHEMAS)}")
        self.st = SafeTensorsFile(str(mf))
        if self.schema in INT8_SCHEMAS:
            self.kind, aux, tag = "int8", INT8_AUX, ".weight.int8_state"
        elif self.schema == NF4_SCHEMA:
            self.kind, aux, tag = "nf4", NF4_AUX, ".weight.quant_state.bitsandbytes__nf4"
        else:
            self.kind, aux, tag = "nested-nf8", NESTED_AUX, ".weight.nested_state"
        self._aux = aux
        self.stems = sorted(n[: -len(tag)] for n in self.st.names() if n.endswith(tag))

    def close(self) -> None:
        self.st.close()

    def __enter__(self) -> "Pin":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def owned(self) -> set:
        out = set()
        for s in self.stems:
            out.add(s + ".weight")
            out.update(s + a for a in self._aux + NF4_DQ_AUX if s + a in self.st.tensors)
        if self.kind == "nested-nf8":
            out.add("_nf8_cells")
        return out

    def passthrough(self) -> List[str]:
        own = self.owned()
        return sorted(n for n in self.st.names() if n not in own)

    def state(self, stem: str) -> dict:
        tag = {"int8": ".weight.int8_state", "nf4": ".weight.quant_state.bitsandbytes__nf4",
               "nested-nf8": ".weight.nested_state"}[self.kind]
        return _json_tensor(self.st, stem + tag)

    def check_stem(self, stem: str) -> List[str]:
        """ABI errors for one linear. Empty list = loads."""
        errs: List[str] = []
        t = self.st.tensors
        for a in ("",) + self._aux:
            if stem + (a or ".weight") not in t:
                errs.append(f"{stem}: missing {stem}{a or '.weight'}")
        if errs:
            return errs
        try:
            state = self.state(stem)
        except PinError as e:
            return [str(e)]
        shape = state.get("shape")
        bs = state.get("blocksize")
        if not (isinstance(shape, list) and len(shape) == 2 and all(isinstance(d, int) and d > 0 for d in shape)):
            return [f"{stem}: state.shape={shape!r} is not [out, in]"]
        if not (isinstance(bs, int) and bs > 0):
            return [f"{stem}: state.blocksize={bs!r} is not a positive int"]
        n = shape[0] * shape[1]
        nblk = (n + bs - 1) // bs
        w = t[stem + ".weight"]
        if self.kind == "int8":
            if state.get("quant_type") != "int8_symmetric":
                errs.append(f"{stem}: quant_type={state.get('quant_type')!r} want int8_symmetric")
            if w.dtype != "I8" or tuple(w.shape) != tuple(shape):
                errs.append(f"{stem}.weight: {w.dtype}{list(w.shape)} want I8{shape}")
            sc = t[stem + ".weight.int8_scale"]
            if sc.dtype != "F32" or tuple(sc.shape) != (nblk,):
                errs.append(f"{stem}.weight.int8_scale: {sc.dtype}{list(sc.shape)} want F32[{nblk}]")
            else:
                bad = [v for v in unpack_f32(self.st.read_bytes(stem + ".weight.int8_scale"))
                       if not (v > 0.0 and v < float("inf"))]
                if bad:
                    errs.append(f"{stem}.weight.int8_scale: {len(bad)} non-positive/non-finite scales")
        else:
            want_q = "nf4" if self.kind == "nf4" else "nested_nf8"
            if state.get("quant_type") != want_q:
                errs.append(f"{stem}: quant_type={state.get('quant_type')!r} want {want_q}")
            half = (n + 1) // 2
            if w.dtype != "U8" or w.nbytes != half:
                errs.append(f"{stem}.weight: {w.dtype} {w.nbytes}B want U8 {half}B packed nibbles")
            am = t[stem + ".weight.absmax"]
            dq = am.dtype == "U8" and all(stem + a in t for a in NF4_DQ_AUX)
            if dq:
                if am.nbytes != nblk:
                    errs.append(f"{stem}.weight.absmax: U8 {am.nbytes}B want {nblk}B (double quant)")
            elif am.dtype != "F32" or tuple(am.shape) != (nblk,):
                errs.append(f"{stem}.weight.absmax: {am.dtype}{list(am.shape)} want F32[{nblk}]")
            qm = t[stem + ".weight.quant_map"]
            if qm.dtype != "F32" or tuple(qm.shape) != (16,):
                errs.append(f"{stem}.weight.quant_map: {qm.dtype}{list(qm.shape)} want F32[16]")
            if self.kind == "nested-nf8":
                pl = t[stem + ".weight.nf8_plug"]
                if pl.dtype != "U8" or pl.nbytes != half:
                    errs.append(f"{stem}.weight.nf8_plug: {pl.dtype} {pl.nbytes}B want U8 {half}B")
        return errs

    def check(self) -> List[str]:
        errs: List[str] = []
        for name, info in self.st.tensors.items():
            if info.dtype not in DTYPE_WIDTH:
                errs.append(f"{name}: unknown dtype {info.dtype}")
            elif info.nbytes != _numel(info.shape) * DTYPE_WIDTH[info.dtype]:
                errs.append(f"{name}: {info.nbytes}B does not match {info.dtype}{list(info.shape)}")
        if not self.stems:
            errs.append(f"no quantized linears for schema {self.schema}")
        for s in self.stems:
            errs.extend(self.check_stem(s))
        if self.kind == "nested-nf8":
            c = self.st.tensors.get("_nf8_cells")
            if c is None:
                errs.append("_nf8_cells missing")
            elif c.dtype != "F32" or tuple(c.shape) != (16, 16):
                errs.append(f"_nf8_cells: {c.dtype}{list(c.shape)} want F32[16,16]")
            else:
                got = unpack_f32(self.st.read_bytes("_nf8_cells"))
                want = unpack_f32(pack_f32(cells_flat()))
                if got != want:
                    errs.append("_nf8_cells differs from nested_nf.NF8_CELLS (do not hand-edit)")
        own = self.owned()
        for n in self.st.names():
            if n in own:
                continue
            if n.endswith(INT8_AUX + NESTED_AUX + NF4_AUX + NF4_DQ_AUX):
                errs.append(f"{n}: orphan aux tensor (no matching linear)")
        n_conv = self.meta.get("n_converted")
        if isinstance(n_conv, int) and self.meta.get("dest") != "nf4" and n_conv != len(self.stems):
            errs.append(f"pin.json n_converted={n_conv} but model has {len(self.stems)} linears")
        return errs

    def dequant(self, stem: str):
        """f32 values of one linear, row-major (numpy f64 array if available, else list)."""
        state = self.state(stem)
        shape, bs = state["shape"], int(state["blocksize"])
        n = int(shape[0]) * int(shape[1])
        raw = self.st.read_bytes(stem + ".weight")
        if self.kind == "int8":
            scales = unpack_f32(self.st.read_bytes(stem + ".weight.int8_scale"))
            if fast.available():
                return fast.dequant_int8(fast.np.frombuffer(raw, dtype=fast.np.int8), scales, bs)
            return dequant_int8(raw, scales, bs)
        if self.st.tensors[stem + ".weight.absmax"].dtype == "U8":
            absmax = decode_absmax(
                self.st.read_bytes(stem + ".weight.absmax"),
                absmax_is_u8=True,
                nested_quant_map=unpack_f32(self.st.read_bytes(stem + ".weight.nested_quant_map")),
                nested_absmax=unpack_f32(self.st.read_bytes(stem + ".weight.nested_absmax")),
                nested_offset=float(state.get("nested_offset") or 0.0),
                nested_blocksize=int(state.get("nested_blocksize") or 256),
            )
        else:
            absmax = unpack_f32(self.st.read_bytes(stem + ".weight.absmax"))
        if self.kind == "nf4":
            cb = unpack_f32(self.st.read_bytes(stem + ".weight.quant_map"))
            if fast.available():
                return fast.dequant_nf4(raw, absmax, n, bs, codebook=cb)
            return dequant_nf4(raw, absmax, n, bs, codebook=cb)
        plug = self.st.read_bytes(stem + ".weight.nf8_plug")
        if fast.available():
            return fast.decode_nested(fast.unpack_nibbles(raw, n), fast.unpack_nibbles(plug, n),
                                      absmax, NF8_CELLS, bs)
        return decode_nested(unpack_indices(raw, n), unpack_indices(plug, n), absmax, bs)


def verify_pin(pin_dir: Path, *, expand: bool = False) -> Dict[str, Any]:
    """Fail-closed ABI report. ok=False lists every error found."""
    try:
        pin = Pin(pin_dir)
    except PinError as e:
        return {"ok": False, "pin": str(pin_dir), "errors": [str(e)]}
    with pin:
        errs = pin.check()
        report: Dict[str, Any] = {
            "ok": not errs,
            "pin": str(pin_dir),
            "schema": pin.schema,
            "kind": pin.kind,
            "n_linears": len(pin.stems),
            "n_passthrough": len(pin.passthrough()),
            "errors": errs,
        }
        if expand and not errs:
            n_elem = 0
            amax = 0.0
            for s in pin.stems:
                v = pin.dequant(s)
                n_elem += len(v)
                if len(v):
                    m = float(abs(v).max()) if fast.available() else max(abs(x) for x in v)
                    amax = max(amax, m)
            report["expanded"] = {"n_elem": n_elem, "max_abs": amax}
    return report
