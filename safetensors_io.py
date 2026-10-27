"""Minimal safetensors reader/writer. Stdlib only. Offline pin use."""
from __future__ import annotations
import json
import os
import struct
from dataclasses import dataclass
from typing import BinaryIO, Dict, Iterable, List, Mapping, Optional, Tuple
DTYPE_WIDTH = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}
@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    data_offsets: Tuple[int, int] # relative to data section
    @property
    def nbytes(self) -> int:
        return int(self.data_offsets[1] - self.data_offsets[0])
class SafeTensorsFile:
    def __init__(self, path: str):
        self.path = path
        self._fp = open(path, "rb")
        header_len = struct.unpack("<Q", self._fp.read(8))[0]
        if header_len > 100_000_000:
            raise ValueError(f"implausible header_len={header_len}")
        raw = self._fp.read(header_len)
        header = json.loads(raw.decode("utf-8"))
        self.metadata = header.pop("__metadata__", {}) or {}
        self.data_start = 8 + header_len
        self.tensors: Dict[str, TensorInfo] = {}
        for name, spec in header.items():
            off = spec["data_offsets"]
            self.tensors[name] = TensorInfo(
                name=name,
                dtype=str(spec["dtype"]),
                shape=tuple(int(x) for x in spec["shape"]),
                data_offsets=(int(off[0]), int(off[1])),
            )
    def close(self) -> None:
        self._fp.close()
    def __enter__(self) -> "SafeTensorsFile":
        return self
    def __exit__(self, *exc) -> None:
        self.close()
    def read_bytes(self, name: str) -> bytes:
        info = self.tensors[name]
        self._fp.seek(self.data_start + info.data_offsets[0])
        return self._fp.read(info.nbytes)
    def names(self) -> List[str]:
        return list(self.tensors.keys())
class SafeTensorsSet:
    """One or more safetensors files as a single name catalog (HF shards)."""
    def __init__(self, files: List[SafeTensorsFile]):
        if not files:
            raise ValueError("SafeTensorsSet needs at least one file")
        self.files = files
        self.path = files[0].path
        self.tensors: Dict[str, TensorInfo] = {}
        self._owner: Dict[str, SafeTensorsFile] = {}
        for f in files:
            for name, info in f.tensors.items():
                if name in self.tensors:
                    raise ValueError(f"duplicate tensor across shards: {name}")
                self.tensors[name] = info
                self._owner[name] = f
    def names(self) -> List[str]:
        return list(self.tensors.keys())
    def read_bytes(self, name: str) -> bytes:
        return self._owner[name].read_bytes(name)
    def close(self) -> None:
        for f in self.files:
            f.close()
    def __enter__(self) -> "SafeTensorsSet":
        return self
    def __exit__(self, *exc) -> None:
        self.close()
def _align8(n: int) -> int:
    return (n + 7) & ~7
def write_safetensors(
    path: str,
    tensors: Iterable[Tuple[str, str, Tuple[int, ...], bytes]],
    metadata: Optional[Mapping[str, str]] = None,
) -> None:
    """Write tensors as (name, dtype, shape, raw_bytes). 8-byte aligned."""
    planned: List[Tuple[str, str, Tuple[int, ...], bytes, int, int]] = []
    cursor = 0
    for name, dtype, shape, blob in tensors:
        if dtype not in DTYPE_WIDTH:
            raise ValueError(f"unsupported dtype {dtype}")
        start = cursor
        end = start + len(blob)
        planned.append((name, dtype, shape, blob, start, end))
        cursor = end
    header: dict = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for name, dtype, shape, _blob, start, end in planned:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, end]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # some readers want 8-aligned header
    pad = (-(len(header_bytes) + 8)) % 8
    if pad:
        header_bytes += b" " * pad
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<Q", len(header_bytes)))
        fp.write(header_bytes)
        pos = 0
        for _name, _dtype, _shape, blob, start, end in planned:
            if pos < start:
                fp.write(b"\x00" * (start - pos))
                pos = start
            fp.write(blob)
            pos = end
        tail = _align8(pos)
        if tail > pos:
            fp.write(b"\x00" * (tail - pos))
