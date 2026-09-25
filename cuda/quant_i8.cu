/* BF16/F32 → symmetric INT8 per block (B=64, zp=0, scale=amax/127).
 * Offline pin convert. Not train. */
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    return __uint_as_float(u);
}
__global__ void k_quant_bf16(const uint16_t* in, int64_t n, int B, int8_t* q,
                             float* scale) {
    int blk = (int)blockIdx.x;
    int t = (int)threadIdx.x;
    int64_t start = (int64_t)blk * (int64_t)B;
    __shared__ float sh[64];
    float a = 0.f;
    float w = 0.f;
    int64_t i = start + (int64_t)t;
    int in_range = (t < B && i < n) ? 1 : 0;
    if (in_range) {
        w = bf16_to_f32(in[i]);
        a = fabsf(w);
    }
    sh[t] = a;
    __syncthreads();
    for (int s = 32; s > 0; s >>= 1) {
        if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]);
        __syncthreads();
    }
    float am = sh[0];
    float s = (am > 0.f) ? (am / 127.f) : 1.f;
    if (t == 0) scale[blk] = s;
    __syncthreads();
    if (in_range) {
        int v = (int)rintf(w / s);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
}
__global__ void k_quant_f32(const float* in, int64_t n, int B, int8_t* q,
                            float* scale) {
    int blk = (int)blockIdx.x;
    int t = (int)threadIdx.x;
    int64_t start = (int64_t)blk * (int64_t)B;
    __shared__ float sh[64];
    float a = 0.f;
    float w = 0.f;
    int64_t i = start + (int64_t)t;
    int in_range = (t < B && i < n) ? 1 : 0;
    if (in_range) {
        w = in[i];
        a = fabsf(w);
    }
    sh[t] = a;
    __syncthreads();
    for (int s = 32; s > 0; s >>= 1) {
        if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]);
        __syncthreads();
    }
    float am = sh[0];
    float s = (am > 0.f) ? (am / 127.f) : 1.f;
    if (t == 0) scale[blk] = s;
    __syncthreads();
    if (in_range) {
        int v = (int)rintf(w / s);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
}
static int launch(const void* h_in, size_t in_bytes, int64_t n, int B, int bf16,
                  int8_t* h_q, float* h_scale) {
    if (B != 64 || n < 1) return 2;
    int64_t nblk = (n + B - 1) / B;
    void* d_in = NULL;
    int8_t* d_q = NULL;
    float* d_s = NULL;
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess) return 3;
    if (cudaMalloc(&d_q, (size_t)n) != cudaSuccess) {
        cudaFree(d_in);
        return 3;
    }
    if (cudaMalloc(&d_s, (size_t)nblk * sizeof(float)) != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_q);
        return 3;
    }
    if (cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_q);
        cudaFree(d_s);
        return 4;
    }
    if (bf16)
        k_quant_bf16<<<(unsigned)nblk, 64>>>((const uint16_t*)d_in, n, B, d_q, d_s);
    else
        k_quant_f32<<<(unsigned)nblk, 64>>>((const float*)d_in, n, B, d_q, d_s);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_q);
        cudaFree(d_s);
        return 5;
    }
    int rc = 0;
    if (cudaMemcpy(h_q, d_q, (size_t)n, cudaMemcpyDeviceToHost) != cudaSuccess) rc = 4;
    if (cudaMemcpy(h_scale, d_s, (size_t)nblk * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        rc = 4;
    cudaFree(d_in);
    cudaFree(d_q);
    cudaFree(d_s);
    return rc;
}
extern "C" int quant_bf16_i8_block(const uint16_t* h_in, int64_t n, int B,
                                   int8_t* h_q, float* h_scale) {
    return launch(h_in, (size_t)n * 2u, n, B, 1, h_q, h_scale);
}
extern "C" int quant_f32_i8_block(const float* h_in, int64_t n, int B, int8_t* h_q,
                                  float* h_scale) {
    return launch(h_in, (size_t)n * 4u, n, B, 0, h_q, h_scale);
}
/* Dequant error vs original BF16: recon = i8 * scale. Not train. */
__global__ void k_err_bf16(const uint16_t* in, const int8_t* q, const float* scale,
                           int64_t n, int B, double* blk_ss, float* blk_mx,
                           unsigned int* blk_over) {
    int blk = (int)blockIdx.x;
    int t = (int)threadIdx.x;
    int64_t start = (int64_t)blk * (int64_t)B;
    int64_t i = start + (int64_t)t;
    __shared__ float sh_max[64];
    __shared__ double sh_ss[64];
    __shared__ unsigned int sh_ov[64];
    float e = 0.f;
    double ss = 0.0;
    unsigned int ov = 0;
    if (t < B && i < n) {
        float s = scale[blk];
        float w = bf16_to_f32(in[i]);
        float d = (float)q[i] * s;
        e = fabsf(w - d);
        ss = (double)e * (double)e;
        float half = 0.5f * s;
        float slack = half * 1.0000002f + 1e-6f;
        if (e > slack) ov = 1u;
    }
    sh_max[t] = e;
    sh_ss[t] = ss;
    sh_ov[t] = ov;
    __syncthreads();
    for (int k = 32; k > 0; k >>= 1) {
        if (t < k) {
            sh_max[t] = fmaxf(sh_max[t], sh_max[t + k]);
            sh_ss[t] += sh_ss[t + k];
            sh_ov[t] += sh_ov[t + k];
        }
        __syncthreads();
    }
    if (t == 0) {
        blk_ss[blk] = sh_ss[0];
        blk_mx[blk] = sh_max[0];
        blk_over[blk] = sh_ov[0];
    }
}
extern "C" int compare_bf16_i8_block(const uint16_t* h_in, const int8_t* h_q,
                                     const float* h_scale, int64_t n, int B,
                                     double* out_sum_sq, float* out_max_abs,
                                     unsigned long long* out_n_over) {
    if (B != 64 || n < 1 || !h_in || !h_q || !h_scale || !out_sum_sq ||
        !out_max_abs || !out_n_over)
        return 2;
    int64_t nblk = (n + B - 1) / B;
    uint16_t* d_in = NULL;
    int8_t* d_q = NULL;
    float* d_s = NULL;
    double* d_ss = NULL;
    float* d_mx = NULL;
    unsigned int* d_ov = NULL;
    int rc = 0;
    if (cudaMalloc(&d_in, (size_t)n * 2u) != cudaSuccess) return 3;
    if (cudaMalloc(&d_q, (size_t)n) != cudaSuccess) {
        cudaFree(d_in);
        return 3;
    }
    if (cudaMalloc(&d_s, (size_t)nblk * sizeof(float)) != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_q);
        return 3;
    }
    if (cudaMalloc(&d_ss, (size_t)nblk * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_mx, (size_t)nblk * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_ov, (size_t)nblk * sizeof(unsigned int)) != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_q);
        cudaFree(d_s);
        cudaFree(d_ss);
        cudaFree(d_mx);
        cudaFree(d_ov);
        return 3;
    }
    if (cudaMemcpy(d_in, h_in, (size_t)n * 2u, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(d_q, h_q, (size_t)n, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_s, h_scale, (size_t)nblk * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        rc = 4;
    } else {
        k_err_bf16<<<(unsigned)nblk, 64>>>(d_in, d_q, d_s, n, B, d_ss, d_mx, d_ov);
        if (cudaDeviceSynchronize() != cudaSuccess) rc = 5;
    }
    double* h_ss = NULL;
    float* h_mx = NULL;
    unsigned int* h_ov = NULL;
    if (rc == 0) {
        h_ss = (double*)malloc((size_t)nblk * sizeof(double));
        h_mx = (float*)malloc((size_t)nblk * sizeof(float));
        h_ov = (unsigned int*)malloc((size_t)nblk * sizeof(unsigned int));
        if (!h_ss || !h_mx || !h_ov) rc = 3;
    }
    if (rc == 0) {
        if (cudaMemcpy(h_ss, d_ss, (size_t)nblk * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(h_mx, d_mx, (size_t)nblk * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(h_ov, d_ov, (size_t)nblk * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            rc = 4;
    }
    if (rc == 0) {
        double ss = 0.0;
        float mx = 0.f;
        unsigned long long ov = 0;
        for (int64_t b = 0; b < nblk; b++) {
            ss += h_ss[b];
            if (h_mx[b] > mx) mx = h_mx[b];
            ov += (unsigned long long)h_ov[b];
        }
        *out_sum_sq = ss;
        *out_max_abs = mx;
        *out_n_over = ov;
    }
    free(h_ss);
    free(h_mx);
    free(h_ov);
    cudaFree(d_in);
    cudaFree(d_q);
    cudaFree(d_s);
    cudaFree(d_ss);
    cudaFree(d_mx);
    cudaFree(d_ov);
    return rc;
}
