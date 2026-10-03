/* nested_nf8.h — load ABI for nested_nf8_pin_v1
 * Hole = NF4 nibble (existing qpack). Plug = 4-bit sub-index in that cell.
 * W = NF8_CELLS[hole][plug] * absmax[i/blocksize]
 * L0: expand to f32 at load. L1: H-TILE lookup instead of codebook[nibble].
 * Generated cells: python3 nested_nf.py --header include/nf8_cells.h
 */
#ifndef ORCH_NESTED_NF8_H_
#define ORCH_NESTED_NF8_H_
#include "nf8_cells.h"
#include <stdint.h>
#define DTYPE_NESTED_NF8 7 /* tensor_meta: 5=nf4, 6 reserved int8 pin, 7=nested_nf8 */
typedef struct {
    int32_t blocksize; /* 64 */
    int32_t nibble_order; /* 0=lo_then_hi */
    int32_t out_features;
    int32_t in_features;
    const uint8_t *hole; /* packed NF4, numel/2 */
    const uint8_t *plug; /* packed 4-bit sub-index, numel/2 */
    const float *absmax; /* n_elem / blocksize */
} NestedNf8Meta;
static inline float nested_nf8_dequant_one(uint8_t hole_nibble, uint8_t plug_nibble, float absmax) {
    unsigned h = (unsigned)(hole_nibble & 15u);
    unsigned p = (unsigned)(plug_nibble & 15u);
    return NF8_CELLS[h][p] * absmax;
}
#endif
