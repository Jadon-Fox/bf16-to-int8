# INT8 pin ABI v1
Default source is **dense BF16/F16**. Dest:
| `--to` | schema | orch |
|---|---|---|
| `int8` | `bf16_to_int8_pin_v1` (legacy `nf4_to_int8_pin_v1` still loads) | INT8 pin loader L0 |
| `nf4` | `bf16_to_nf4_pin_v1` | fit, owned NF4 |
| `nested-nf8` | `nested_nf8_pin_v1` | hole=NF4 nibble, plug=4-bit sub-cell. `w=NF8_CELLS[h][p]*absmax` |
NF4 sources are HARD_BLOCK without `--allow-requant`. `nested-nf8` **never** from NF4 (no within-cell bits).
## nested_nf8_pin_v1
Per linear:
- `{stem}.weight` U8 packed hole (NF4 nibbles, lo-then-hi)
- `{stem}.weight.nf8_plug` U8 packed plug (4-bit sub-index)
- `{stem}.weight.absmax` F32
- `{stem}.weight.quant_map` F32 16 (parent NF4 table)
- `{stem}.weight.nested_state` JSON
- `_nf8_cells` F32 [16,16] once per file (SSOT also `include/nf8_cells.h`)
## Directory
```
<pin>/
  model.safetensors
  pin.json
  CONVERT_REPORT.json
  config.json # quantization_config.quant_method = bf16_to_int8_pin
  tokenizer* / generation_* # copied if present
```
`pin.json` (required):
```json
{
  "schema": "bf16_to_int8_pin_v1",
  "n_nf4_modules": 122,
  "n_passthrough": 72,
  "int8_blocksize": 64,
  "int8_scheme": "symmetric_per_block_zp0"
}
```
Loader requires `schema=bf16_to_int8_pin_v1` (or legacy `nf4_to_int8_pin_v1`). Extra keys are informational:
```json
{
  "schema": "bf16_to_int8_pin_v1",
  "n_nf4_modules": 122,
  "n_dense_modules": 6,
  "n_converted": 128,
  "n_passthrough": 72,
  "src_kinds": ["nf4", "bf16"],
  "policy": {"linears": "int8", "dense": "int8", "embed": "copy", "norm": "copy"},
  "int8_blocksize": 64,
  "int8_scheme": "symmetric_per_block_zp0"
}
```
`src_quant` in each `int8_state` may be `nf4`, `fp4`, `bf16`, `f16`, or `f32`. Dequant formula is the same.
**Always copied (never INT8):** RMSNorm, LayerNorm, bias, rotary `inv_freq`.
## Per converted linear (was NF4)
| Tensor | dtype | shape | bytes |
|--------|-------|-------|-------|
| `{stem}.weight` | **I8** | `[out, in]` | `out*in` signed bytes, two's complement, row-major |
| `{stem}.weight.int8_scale` | **F32** LE | `[ceil(out*in / B)]` | `scale = amax/127` |
| `{stem}.weight.int8_state` | **U8** | `[nbytes]` | UTF-8 JSON, not pickle |
`int8_state` JSON:
```json
{
  "quant_type": "int8_symmetric",
  "blocksize": 64,
  "shape": [out, in],
  "src_quant": "nf4",
  "double_quant": "double"
}
```
`double_quant` is `"double"` or `"single"` (source NF4). Loader does **not** re-decode NF4.
### Dequant (host or device, once at load or per tile)
```
B = blocksize (64)
w[i] = int8_as_signed(weight_bytes[i]) * scale[i / B]
int8_as_signed: b < 128 ? b : b - 256
qmin=-127 qmax=127 zp=0
```
Phi-4 Mini stems (122): all `self_attn.qkv_proj` / `o_proj`; `mlp.gate_up_proj` / `down_proj` except layers **1, 3, 30**.
## Passthrough (copy, do not requant)
Embed, RMSNorms, and skipped MLPs stay **BF16** (or whatever dtype they had). Names unchanged.
## C side-POD (add; do not stuff into TensorMeta)
Existing `DTYPE_NF4 = 5`. Suggest:
```
#define DTYPE_INT8 6
typedef struct {
    int32_t blocksize; /* 64 */
    int32_t out_features;
    int32_t in_features;
    int32_t zp; /* 0 */
} Int8Meta;
```
`TensorMeta.dtype = DTYPE_INT8` → `ptr` is I8 packed row-major. Scales are a **companion** F32 buffer, length `ceil(numel/blocksize)`, not inside `TensorMeta`.
## Loader must not
- Invent `ORCH_BASE_PACK`
- Default the product pin off NF4
- Claim INT8 Tensor-Core MMA from this file (storage only)
- Convert per step
