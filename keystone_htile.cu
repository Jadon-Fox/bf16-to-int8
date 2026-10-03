/* Keystone H-TILE sketch — CUDA only, Ampere sm_86.
 * NF4 codebook replaced by bitwise hole|plug inflate.
 * Plug[L] is already in VRAM (stream-1 memcpy of L+1 overlaps).
 * Inner-loop memcpy forbidden.
 *
 * nvcc -arch=sm_86 -c keystone_htile.cu
 */
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef KS_HTILE_M
#define KS_HTILE_M 64
#define KS_HTILE_N 64
#define KS_HTILE_K 64
#endif
__device__ __forceinline__ __nv_bfloat16 inflate_u16(uint16_t bits) {
    return *reinterpret_cast<const __nv_bfloat16 *>(&bits);
}
/* hole: 4 MSBs packed lo-then-hi (orch nibble walk).
 * plug: uint16 low 12 bits, N*K.
 * x: bf16 M*K. y: f32 M*N.
 * S1 FMA body — S5 swaps the kk loop for mma.sync.bf16.
 */
__global__ void keystone_htile_s1_f32(
    const uint8_t *__restrict__ hole,
    const uint16_t *__restrict__ plug,
    const __nv_bfloat16 *__restrict__ x,
    float *__restrict__ y,
    int M, int N, int K)
{
    const int n0 = blockIdx.x * KS_HTILE_N;
    const int m0 = blockIdx.y * KS_HTILE_M;
    const int tn = threadIdx.x;
    const int tm = threadIdx.y;
    __shared__ uint8_t sh_hi4[KS_HTILE_N * KS_HTILE_K];
    __shared__ uint16_t sh_plug[KS_HTILE_N * KS_HTILE_K];
    __shared__ __nv_bfloat16 sh_x[KS_HTILE_M * KS_HTILE_K];
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += KS_HTILE_K) {
        const int n_idx = n0 + tn;
        const int k_idx = k0 + tm;
        if (n_idx < N && k_idx < K) {
            const int w = n_idx * K + k_idx;
            const int tlin = tn * KS_HTILE_K + tm;
            sh_plug[tlin] = plug[w] & 0x0FFFu;
            const uint8_t b = hole[w >> 1];
            sh_hi4[tlin] = (w & 1) ? (uint8_t)((b >> 4) & 0xF) : (uint8_t)(b & 0xF);
        }
        const int m_idx = m0 + tn;
        if (m_idx < M && k_idx < K)
            sh_x[tn * KS_HTILE_K + tm] = x[m_idx * K + k_idx];
        __syncthreads();
        if ((m0 + tm) < M && (n0 + tn) < N) {
            for (int kk = 0; kk < KS_HTILE_K && (k0 + kk) < K; ++kk) {
                const int tlin = tn * KS_HTILE_K + kk;
                const uint16_t bits =
                    (uint16_t)((sh_hi4[tlin] << 12) | (sh_plug[tlin] & 0x0FFF));
                acc += __bfloat162float(inflate_u16(bits)) *
                       __bfloat162float(sh_x[tm * KS_HTILE_K + kk]);
            }
        }
        __syncthreads();
    }
    if ((m0 + tm) < M && (n0 + tn) < N)
        y[(m0 + tm) * N + (n0 + tn)] = acc;
}
/* S3: cp.async D=2 on hole + plug (same commit/wait as orch H-TILE).
 * S5: mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32 on inflated fragments.
 * Host: cudaMemcpyAsync(plug[L+1], stream1) while this runs on stream0.
 */
