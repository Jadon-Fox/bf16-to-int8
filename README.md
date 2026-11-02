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
## Block INT8 quant backends (pin convert)
Dense 16-bit linears use **symmetric block INT8** with `blocksize=64` (`nf4.quantize_int8_symmetric` in Python, or GPU when built). The pin converter (`pin_convert.py`) calls **`gpu_quant.quant_bf16_i8`** only; it does not load the Mojo module.
| Path | Artifact | Role |
|------|----------|------|
| CUDA | `bash cuda/build_quant_i8.sh` → `cuda/libquant_i8.so` | **Production** GPU quant for pin convert (sm_86) |
| Mojo twin | `bash mojo/build.sh` → `mojo/quant_i8_mojo.so` | Same ABI as CUDA; catches Mojo API drift without a GPU at build time (`MOJO_ACCEL`, default `sm_86`) |
| Python | `nf4.py` | Fallback when no `.so` is present |
Build the Mojo twin with `pip install mojo max` (CI pins Mojo 1.1 / MAX 26.6). `mojo_quant.py` exposes `quant_bf16_i8` / `quant_f32_i8` with `device=cpu|gpu`; missing build returns `None` like `gpu_quant`.
**Twin check (not pin convert):**
```bash
bash cuda/build_quant_i8.sh && bash mojo/build.sh
python -m pytest -q test_mojo_quant.py          # CPU golden vs nf4 golden; skipped if .so missing
python3 compare_mojo_vs_cuda.py --n $((1<<20))  # JSON report; exit 1 on bound or ref mismatch
```
Advisory CI: `.github/workflows/mojo.yml` (cross-compile + CPU golden; `continue-on-error: true`).
