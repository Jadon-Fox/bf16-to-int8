# convert pin — BF16 → INT8 (or NF4)
**Download BF16 (or F16). Convert once.** Do not download GGUF/GPTQ/AWQ/Unsloth-4bit and requant.
Repo name hangover was `nf4-to-int8`. Canonical dest schema is `bf16_to_int8_pin_v1`. NF4→INT8 is a lossy second hop (`--allow-requant` only).
```bash
# default: BF16/F16 safetensors → INT8 pin (orch loader ABI)
python3 bf16_to_int8.py pin --src /path/to/phi-4-mini-bf16 --out /path/to/int8-pin --to int8
# hangover CLI name (same entry):
python3 nf4_to_int8.py pin --src /path/to/phi-4-mini-bf16 --out /path/to/int8-pin --to int8
# fit case: same source → NF4 pin
python3 nf4_to_int8.py pin --src /path/to/phi-4-mini-bf16 --out /path/to/nf4-pin --to nf4
# nested NF8: keep the 16 Gaussian cells, plug = sub-index (orch H-TILE L0/L1)
python3 nf4_to_int8.py pin --src /path/to/phi-4-mini-bf16 --out /path/to/nested-nf8 --to nested-nf8
python3 nf4_to_int8.py pin --src model.safetensors --out /tmp/pin --dry-run
# HF snapshots with model.safetensors.index.json (2+ shards) are read as one catalog.
```
Check a pin before handing it to orch (CPU, no source needed, exit 3 on any ABI error):
```bash
python3 bf16_to_int8.py verify --pin /path/to/int8-pin           # schema, shapes, scales, state JSON
python3 bf16_to_int8.py verify --pin /path/to/int8-pin --expand  # also dequant every linear on the host
python3 compare_pin_vs_src.py --pin /path/to/int8-pin             # vs original BF16. CUDA if built, else CPU
```
Stdlib only by default. If `numpy` is installed the quant loops run vectorized and write **the same bytes** (tested). `BF16_TO_INT8_NO_NUMPY=1` forces the stdlib path. Output `model.safetensors` follows the spec (contiguous offsets), so HF `safetensors` opens it.
Already-quantized NF4 is **refused** unless `--allow-requant` (lossy second hop; cannot restore BF16).
| flag | default | |
|------|---------|--|
| `--to` | `int8` | dest pin: `int8` or `nf4` |
| `--dense` | `quantize` | 16-bit linears → dest. `copy` to leave BF16 |
| `--embed` | `copy` | embeddings / lm_head stay BF16 |
| (norms) | always copy | RMSNorm / bias / rotary never quantized |
| `--allow-requant` | off | only if the only copy you have is already NF4 |
GPTQ, AWQ, GGUF: refuse. Get the HuggingFace **BF16/F16** tree.
## Why both dests
| dest | when |
|------|------|
| **INT8** | better unfold from BF16, ~8 bit, orch `bf16_to_int8_pin_v1` loader |
| **NF4** | need to **fit** on 12 GB. Better 4-bit than INT4 for QLoRA-style dequant. Schema `bf16_to_nf4_pin_v1` |
Ampere still GEMMs in f16/f32. Storage only. Meaning Version: 0.3.4.
INT4 dest is not offered (worse unfold than NF4 at the same size).
