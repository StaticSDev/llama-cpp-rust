// Self-contained CUDA dequant/matvec library for
// Qwen3.5-4B-Q4_K_M.gguf style quantized tensors.
// Target: NVIDIA GTX 1660 SUPER (TU116, sm_75), CUDA 13.4.
// Dequant math copied from ggml-cuda/dequantize.cuh and ggml-common.h;
// no ggml headers are included.

#include "lcr_kernels.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

enum {
    LCR_OK        = 0,
    LCR_ERR_ARG   = -1,
    LCR_ERR_CUDA  = -2,
    LCR_ERR_TYPE  = -3,
    LCR_ERR_RANGE = -4,
    LCR_ERR_MEM   = -5
};

// GGML type ids (subset present in Qwen3.5-4B-Q4_K_M.gguf).
enum {
    LCR_F32  = 0,
    LCR_F16  = 1,
    LCR_Q8_0 = 8,
    LCR_Q4_K = 12,
    LCR_Q5_K = 13,
    LCR_Q6_K = 14,
    LCR_BF16 = 30
};

#define LCR_QK_K 256
#define LCR_K_SCALE_SIZE 12
#define LCR_BLOCK_SIZE 256

// GGML block layouts. The structs mirror ggml-common.h exactly.
struct LcrBlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(LcrBlockQ8_0) == 34, "q8_0 block size");

struct LcrBlockQ4_K {
    uint32_t dm; // two ggml_half values: d, dmin
    uint8_t scales[LCR_K_SCALE_SIZE];
    uint8_t qs[LCR_QK_K / 2];
};
static_assert(sizeof(LcrBlockQ4_K) == 144, "q4_K block size");

struct LcrBlockQ5_K {
    uint32_t dm;
    uint8_t scales[LCR_K_SCALE_SIZE];
    uint8_t qh[LCR_QK_K / 8];
    uint8_t qs[LCR_QK_K / 2];
};
static_assert(sizeof(LcrBlockQ5_K) == 176, "q5_K block size");

struct LcrBlockQ6_K {
    uint8_t ql[LCR_QK_K / 2];
    uint8_t qh[LCR_QK_K / 4];
    int8_t scales[LCR_QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(LcrBlockQ6_K) == 210, "q6_K block size");

__device__ __forceinline__ float lcr_half2float(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ float lcr_bf162float(uint16_t b) {
    return __bfloat162float(__ushort_as_bfloat16(b));
}

// ggml get_scale_min_k4, byte-for-byte equivalent.
__device__ __forceinline__ void lcr_get_scale_min_k4(int j, const uint8_t *q,
                                                     uint8_t &d, uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequantize element n of one row. Works for any thread layout: callers
// iterate over n directly, so no ggml-specific thread mapping is needed.
__device__ __forceinline__ float lcr_dequant_at(int type, const uint8_t *row, int n) {
    switch (type) {
    case LCR_F32: {
        float v;
        memcpy(&v, row + (size_t)n * sizeof(float), sizeof(v));
        return v;
    }
    case LCR_F16: {
        uint16_t h;
        memcpy(&h, row + (size_t)n * sizeof(uint16_t), sizeof(h));
        return lcr_half2float(h);
    }
    case LCR_BF16: {
        uint16_t b;
        memcpy(&b, row + (size_t)n * sizeof(uint16_t), sizeof(b));
        return lcr_bf162float(b);
    }
    case LCR_Q8_0: {
        const LcrBlockQ8_0 *blk =
            (const LcrBlockQ8_0 *)(row + (size_t)(n >> 5) * sizeof(LcrBlockQ8_0));
        return (float)blk->qs[n & 31] * lcr_half2float(blk->d);
    }
    case LCR_Q4_K: {
        const LcrBlockQ4_K *blk =
            (const LcrBlockQ4_K *)(row + (size_t)(n >> 8) * sizeof(LcrBlockQ4_K));
        const int j = n & 255;
        const int il = j >> 6;            // 0..3
        const int is = 2 * il + ((j >> 5) & 1);
        uint8_t sc, m;
        lcr_get_scale_min_k4(is, blk->scales, sc, m);
        const float dall = lcr_half2float((uint16_t)(blk->dm & 0xFFFF));
        const float dmin = lcr_half2float((uint16_t)(blk->dm >> 16));
        const uint8_t q = blk->qs[32 * il + (j & 31)];
        const float d = dall * sc;
        const float dm = dmin * m;
        return (j & 32) ? d * (q >> 4) - dm : d * (q & 0xF) - dm;
    }
    case LCR_Q5_K: {
        const LcrBlockQ5_K *blk =
            (const LcrBlockQ5_K *)(row + (size_t)(n >> 8) * sizeof(LcrBlockQ5_K));
        const int j = n & 255;
        const int il = j >> 6;            // 0..3
        const int t = j & 63;
        const int upper = t >> 5;
        const int is = 2 * il + upper;
        uint8_t sc, m;
        lcr_get_scale_min_k4(is, blk->scales, sc, m);
        const float dall = lcr_half2float((uint16_t)(blk->dm & 0xFFFF));
        const float dmin = lcr_half2float((uint16_t)(blk->dm >> 16));
        const int qidx = 32 * il + (t & 31);
        const uint8_t qhbit = (uint8_t)(1u << (2 * il + upper));
        const uint8_t qh = blk->qh[t & 31];
        const uint8_t ql = blk->qs[qidx];
        const int q = (upper ? (ql >> 4) : (ql & 0xF)) | ((qh & qhbit) ? 16 : 0);
        return dall * sc * q - dmin * m;
    }
    case LCR_Q6_K: {
        const LcrBlockQ6_K *blk =
            (const LcrBlockQ6_K *)(row + (size_t)(n >> 8) * sizeof(LcrBlockQ6_K));
        const int j = n & 255;
        const int ip = j >> 7;            // 0..1
        const int t = j & 127;
        const int group = t >> 5;         // 0..3
        const int l = t & 31;
        const int is = 8 * ip + (l >> 4) + 2 * group;
        const uint8_t ql = blk->ql[64 * ip + l + 32 * (group & 1)];
        const uint8_t qh = blk->qh[32 * ip + l] >> (2 * group);
        const uint8_t q4 = (group & 2) ? (ql >> 4) : (ql & 0xF);
        const int q = (int)((int8_t)(q4 | ((qh & 3) << 4)) - 32);
        return lcr_half2float(blk->d) * (float)blk->scales[is] * (float)q;
    }
    default:
        return 0.0f;
    }
}

// --- type-specialized matvec kernels ----------------------------------------
// One CUDA block per output row. For K-quants each warp owns one 256-element
// quant block; its eight warps cover eight blocks per iteration, so all 256
// threads contribute. Four elements per lane keep q and x loads vectorized;
// group scales are read once per subgroup and broadcast. No type branch remains
// inside the element loop.

__device__ __forceinline__ void lcr_matvec_reduce(float *smem, float sum,
                                                  float *out, int row) {
    for (int s = 16; s > 0; s >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, s);
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        smem[wid] = sum;
    }
    __syncthreads();
    if (wid == 0) {
        float v = lane < (LCR_BLOCK_SIZE / 32) ? smem[lane] : 0.0f;
        for (int s = (LCR_BLOCK_SIZE / 32) / 2; s > 0; s >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFFu, v, s);
        }
        if (lane == 0) {
            out[row] = v;
        }
    }
}

__device__ __forceinline__ bool lcr_vec_ok(const void *w, const void *x, int align) {
    return ((uintptr_t)w & (align - 1)) == 0 &&
           ((uintptr_t)x & (align - 1)) == 0;
}

__global__ void lcr_matvec_f32(const uint8_t * __restrict__ blob, int in,
                               const float * __restrict__ x,
                               float * __restrict__ out, int64_t offset,
                               size_t row_bytes) {
    __shared__ float smem[LCR_BLOCK_SIZE / 32];
    const int row = blockIdx.x;
    const float * __restrict__ w =
        (const float *)(blob + offset + (size_t)row * row_bytes);
    float sum = 0.0f;
    if (((in & 3) == 0) && lcr_vec_ok(w, x, 16)) {
        for (int n = threadIdx.x * 4; n + 4 <= in; n += LCR_BLOCK_SIZE * 4) {
            const float4 wv = __ldg((const float4 *)(w + n));
            const float4 xv = __ldg((const float4 *)(x + n));
            sum += (wv.x * xv.x + wv.y * xv.y) + (wv.z * xv.z + wv.w * xv.w);
        }
    } else {
        for (int n = threadIdx.x; n < in; n += LCR_BLOCK_SIZE) {
            sum += __ldg(w + n) * __ldg(x + n);
        }
    }
    lcr_matvec_reduce(smem, sum, out, row);
}

__global__ void lcr_matvec_f16(const uint8_t * __restrict__ blob, int in,
                               const float * __restrict__ x,
                               float * __restrict__ out, int64_t offset,
                               size_t row_bytes) {
    __shared__ float smem[LCR_BLOCK_SIZE / 32];
    const int row = blockIdx.x;
    const uint16_t * __restrict__ w =
        (const uint16_t *)(blob + offset + (size_t)row * row_bytes);
    float sum = 0.0f;
    if (((in & 3) == 0) && lcr_vec_ok(w, x, 8)) {
        for (int n = threadIdx.x * 4; n + 4 <= in; n += LCR_BLOCK_SIZE * 4) {
            const uint2 wv = __ldg((const uint2 *)(w + n));
            const float4 xv = __ldg((const float4 *)(x + n));
            sum += lcr_half2float((uint16_t)wv.x) * xv.x;
            sum += lcr_half2float((uint16_t)(wv.x >> 16)) * xv.y;
            sum += lcr_half2float((uint16_t)wv.y) * xv.z;
            sum += lcr_half2float((uint16_t)(wv.y >> 16)) * xv.w;
        }
    } else {
        for (int n = threadIdx.x; n < in; n += LCR_BLOCK_SIZE) {
            sum += lcr_half2float(__ldg(w + n)) * __ldg(x + n);
        }
    }
    lcr_matvec_reduce(smem, sum, out, row);
}

__global__ void lcr_matvec_bf16(const uint8_t * __restrict__ blob, int in,
                                const float * __restrict__ x,
                                float * __restrict__ out, int64_t offset,
                                size_t row_bytes) {
    __shared__ float smem[LCR_BLOCK_SIZE / 32];
    const int row = blockIdx.x;
    const uint16_t * __restrict__ w =
        (const uint16_t *)(blob + offset + (size_t)row * row_bytes);
    float sum = 0.0f;
    if (((in & 3) == 0) && lcr_vec_ok(w, x, 8)) {
        for (int n = threadIdx.x * 4; n + 4 <= in; n += LCR_BLOCK_SIZE * 4) {
            const uint2 wv = __ldg((const uint2 *)(w + n));
            const float4 xv = __ldg((const float4 *)(x + n));
            sum += lcr_bf162float((uint16_t)wv.x) * xv.x;
            sum += lcr_bf162float((uint16_t)(wv.x >> 16)) * xv.y;
            sum += lcr_bf162float((uint16_t)wv.y) * xv.z;
            sum += lcr_bf162float((uint16_t)(wv.y >> 16)) * xv.w;
        }
    } else {
        for (int n = threadIdx.x; n < in; n += LCR_BLOCK_SIZE) {
            sum += lcr_bf162float(__ldg(w + n)) * __ldg(x + n);
        }
    }
    lcr_matvec_reduce(smem, sum, out, row);
}

__global__ void lcr_matvec_q8_0(const uint8_t * __restrict__ blob, int in,
                                const float * __restrict__ x,
                                float * __restrict__ out, int64_t offset,
                                size_t row_bytes) {
    __shared__ float smem[LCR_BLOCK_SIZE / 32];
    const int row = blockIdx.x;
    const uint8_t * __restrict__ w = blob + offset + (size_t)row * row_bytes;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blocks = in >> 5;
    float sum = 0.0f;
    for (int b = wid; b < blocks; b += LCR_BLOCK_SIZE / 32) {
        const LcrBlockQ8_0 *blk =
            (const LcrBlockQ8_0 *)(w + (size_t)b * sizeof(LcrBlockQ8_0));
        float d = 0.0f;
        if (lane == 0) {
            d = lcr_half2float(__ldg(&blk->d));
        }
        d = __shfl_sync(0xFFFFFFFFu, d, 0);
        const int q = (int)__ldg(&blk->qs[lane]);
        sum += ((float)q * d) * __ldg(x + (size_t)b * 32 + lane);
    }
    lcr_matvec_reduce(smem, sum, out, row);
}

// One warp owns one 256-element quant block. Each lane walks the eight
// 32-element subgroups of that block, so each packed byte is read once and a
// scale/min pair is computed once per subgroup. The CTA's eight warps cover
// eight distinct quant blocks per iteration; all 256 threads contribute.
__global__ void lcr_matvec_q4_k(const uint8_t * __restrict__ blob, int in,
                                const float * __restrict__ x,
                                float * __restrict__ out, int64_t offset,
                                size_t row_bytes) {
    __shared__ float smem[LCR_BLOCK_SIZE / 32];
    const int row = blockIdx.x;
    const uint8_t * __restrict__ w = blob + offset + (size_t)row * row_bytes;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blocks = in >> 8;
    const bool vx = (((uintptr_t)x) & 15) == 0;
    float sum = 0.0f;
    for (int b = wid; b < blocks; b += LCR_BLOCK_SIZE / 32) {
        const LcrBlockQ4_K *blk =
            (const LcrBlockQ4_K *)(w + (size_t)b * sizeof(LcrBlockQ4_K));
        const uint32_t md = __ldg((const uint32_t *)&blk->dm);
        const float dall = lcr_half2float((uint16_t)md);
        const float dmin = lcr_half2float((uint16_t)(md >> 16));
        float acc = 0.0f;
        if (vx) {
            // Eight lanes per 32-element subgroup: one uint32 q load and one
            // float4 x load per four elements.
            const int gbase = lane >> 3;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int g = gbase + t * 4;
                float d = 0.0f, dm = 0.0f;
                if ((lane & 7) == 0) {
                    uint8_t sc, m;
                    lcr_get_scale_min_k4(g, blk->scales, sc, m);
                    d = dall * (float)sc;
                    dm = dmin * (float)m;
                }
                d = __shfl_sync(0xFFFFFFFFu, d, 0, 8);
                dm = __shfl_sync(0xFFFFFFFFu, dm, 0, 8);
                const uint32_t q = __ldg((const uint32_t *)
                    (blk->qs + 32 * (g >> 1) + 4 * (lane & 7)));
                const float4 xv = __ldg((const float4 *)
                    (x + (size_t)b * LCR_QK_K + g * 32 + 4 * (lane & 7)));
                if (g & 1) {
                    acc += (d * (float)((q >> 4) & 15) - dm) * xv.x;
                    acc += (d * (float)((q >> 12) & 15) - dm) * xv.y;
                    acc += (d * (float)((q >> 20) & 15) - dm) * xv.z;
                    acc += (d * (float)((q >> 28) & 15) - dm) * xv.w;
                } else {
                    acc += (d * (float)(q & 15) - dm) * xv.x;
                    acc += (d * (float)((q >> 8) & 15) - dm) * xv.y;
                    acc += (d * (float)((q >> 16) & 15) - dm) * xv.z;
                    acc += (d * (float)((q >> 24) & 15) - dm) * xv.w;
                }
            }
        } else {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                float d = 0.0f, dm = 0.0f;
                if (lane == 0) {
                    uint8_t sc, m;
                    lcr_get_scale_min_k4(s, blk->scales, sc, m);
                    d = dall * (float)sc;
                    dm = dmin * (float)m;
                }
                d = __shfl_sync(0xFFFFFFFFu, d, 0);
                dm = __shfl_sync(0xFFFFFFFFu, dm, 0);
                const uint8_t q = __ldg(&blk->qs[32 * (s >> 1) + lane]);
                const float vq = (s & 1) ? (float)(q >> 4) : (float)(q & 0xF);
                acc += (d * vq - dm) * __ldg(x + (size_t)b * LCR_QK_K + s * 32 + lane);
            }
        }
        sum += acc;
    }
    lcr_matvec_reduce(smem, sum, out, row);
}

__global__ void lcr_matvec_q5_k(const uint8_t * __restrict__ blob, int in,
                                const float * __restrict__ x,
                                float * __restrict__ out, int64_t offset,
                                size_t row_bytes) {
    __shared__ float smem[LCR_BLOCK_SIZE / 32];
    const int row = blockIdx.x;
    const uint8_t * __restrict__ w = blob + offset + (size_t)row * row_bytes;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blocks = in >> 8;
    const bool vx = (((uintptr_t)x) & 15) == 0;
    float sum = 0.0f;
    for (int b = wid; b < blocks; b += LCR_BLOCK_SIZE / 32) {
        const LcrBlockQ5_K *blk =
            (const LcrBlockQ5_K *)(w + (size_t)b * sizeof(LcrBlockQ5_K));
        const uint32_t md = __ldg((const uint32_t *)&blk->dm);
        const float dall = lcr_half2float((uint16_t)md);
        const float dmin = lcr_half2float((uint16_t)(md >> 16));
        float acc = 0.0f;
        if (vx) {
            const int gbase = lane >> 3;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int g = gbase + t * 4;
                float d = 0.0f, dm = 0.0f;
                if ((lane & 7) == 0) {
                    uint8_t sc, m;
                    lcr_get_scale_min_k4(g, blk->scales, sc, m);
                    d = dall * (float)sc;
                    dm = dmin * (float)m;
                }
                d = __shfl_sync(0xFFFFFFFFu, d, 0, 8);
                dm = __shfl_sync(0xFFFFFFFFu, dm, 0, 8);
                const uint32_t ql = __ldg((const uint32_t *)
                    (blk->qs + 32 * (g >> 1) + 4 * (lane & 7)));
                const uint32_t qh = __ldg((const uint32_t *)
                    (blk->qh + 4 * (lane & 7)));
                const float4 xv = __ldg((const float4 *)
                    (x + (size_t)b * LCR_QK_K + g * 32 + 4 * (lane & 7)));
                const uint32_t qhmask = 1u << g;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t shift = 8 * i;
                    const uint32_t l4 = (g & 1) ? ((ql >> (shift + 4)) & 15)
                                                : ((ql >> shift) & 15);
                    const int q = (int)l4 | (((qh >> shift) & qhmask) ? 16 : 0);
                    acc += (d * (float)q - dm) *
                           (i == 0 ? xv.x : i == 1 ? xv.y : i == 2 ? xv.z : xv.w);
                }
            }
        } else {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                float d = 0.0f, dm = 0.0f;
                if (lane == 0) {
                    uint8_t sc, m;
                    lcr_get_scale_min_k4(s, blk->scales, sc, m);
                    d = dall * (float)sc;
                    dm = dmin * (float)m;
                }
                d = __shfl_sync(0xFFFFFFFFu, d, 0);
                dm = __shfl_sync(0xFFFFFFFFu, dm, 0);
                const uint8_t qh = __ldg(&blk->qh[lane]);
                const uint8_t ql = __ldg(&blk->qs[32 * (s >> 1) + lane]);
                const int q = ((s & 1) ? (ql >> 4) : (ql & 0xF)) |
                              ((qh & (1u << s)) ? 16 : 0);
                acc += (d * (float)q - dm) *
                       __ldg(x + (size_t)b * LCR_QK_K + s * 32 + lane);
            }
        }
        sum += acc;
    }
    lcr_matvec_reduce(smem, sum, out, row);
}

__global__ void lcr_matvec_q6_k(const uint8_t * __restrict__ blob, int in,
                                const float * __restrict__ x,
                                float * __restrict__ out, int64_t offset,
                                size_t row_bytes) {
    __shared__ float smem[LCR_BLOCK_SIZE / 32];
    const int row = blockIdx.x;
    const uint8_t * __restrict__ w = blob + offset + (size_t)row * row_bytes;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blocks = in >> 8;
    const bool vx = (((uintptr_t)x) & 15) == 0;
    float sum = 0.0f;
    for (int b = wid; b < blocks; b += LCR_BLOCK_SIZE / 32) {
        const LcrBlockQ6_K *blk =
            (const LcrBlockQ6_K *)(w + (size_t)b * sizeof(LcrBlockQ6_K));
        const float d = lcr_half2float(__ldg(&blk->d));
        float acc = 0.0f;
        // Q6_K blocks are 210 bytes, so use 2-byte loads for the 4-byte groups.
        if (vx) {
            const int gbase = lane >> 3;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int g = gbase + t * 4;
                const int ip = g >> 2;
                const int group = g & 3;
                const int is0 = 8 * ip + 2 * group;
                float s0 = 0.0f, s1 = 0.0f;
                if ((lane & 7) == 0) {
                    const uint16_t ss = __ldg((const uint16_t *)(blk->scales + is0));
                    s0 = (float)(int8_t)ss;
                    s1 = (float)(int8_t)(ss >> 8);
                }
                s0 = __shfl_sync(0xFFFFFFFFu, s0, 0, 8);
                s1 = __shfl_sync(0xFFFFFFFFu, s1, 0, 8);
                const uint16_t *qlp = (const uint16_t *)
                    (blk->ql + 64 * ip + 32 * (group & 1) + 4 * (lane & 7));
                const uint16_t *qhp = (const uint16_t *)
                    (blk->qh + 32 * ip + 4 * (lane & 7));
                const uint32_t ql = (uint32_t)__ldg(qlp) | ((uint32_t)__ldg(qlp + 1) << 16);
                const uint32_t qh = (uint32_t)__ldg(qhp) | ((uint32_t)__ldg(qhp + 1) << 16);
                const float4 xv = __ldg((const float4 *)
                    (x + (size_t)b * LCR_QK_K + g * 32 + 4 * (lane & 7)));
                const float sc = (lane & 7) < 4 ? s0 : s1;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t shift = 8 * i;
                    const uint32_t lo = (group & 2) ? ((ql >> (shift + 4)) & 15)
                                                    : ((ql >> shift) & 15);
                    const uint32_t hi = ((qh >> shift) >> (2 * group)) & 3;
                    const int q = (int)((int8_t)(lo | (hi << 4)) - 32);
                    acc += (d * sc * (float)q) *
                           (i == 0 ? xv.x : i == 1 ? xv.y : i == 2 ? xv.z : xv.w);
                }
            }
        } else {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const int ip = s >> 2;
                const int group = s & 3;
                const int is0 = 8 * ip + 2 * group;
                float s0 = 0.0f, s1 = 0.0f;
                if (lane == 0) {
                    const uint16_t ss = __ldg((const uint16_t *)(blk->scales + is0));
                    s0 = (float)(int8_t)ss;
                    s1 = (float)(int8_t)(ss >> 8);
                }
                s0 = __shfl_sync(0xFFFFFFFFu, s0, 0);
                s1 = __shfl_sync(0xFFFFFFFFu, s1, 0);
                const uint8_t ql = __ldg(&blk->ql[64 * ip + 32 * (group & 1) + lane]);
                const uint8_t qh = __ldg(&blk->qh[32 * ip + lane]);
                const uint8_t q4 = (group & 2) ? (ql >> 4) : (ql & 0xF);
                const int q = (int)((int8_t)(q4 | ((qh >> (2 * group) & 3) << 4)) - 32);
                acc += (d * (lane < 16 ? s0 : s1) * (float)q) *
                       __ldg(x + (size_t)b * LCR_QK_K + s * 32 + lane);
            }
        }
        sum += acc;
    }
    lcr_matvec_reduce(smem, sum, out, row);
}

// One switch per launch, never per element. Type is validated by caller.
static void lcr_launch_matvec(cudaStream_t stream, const uint8_t *blob, int type,
                              int in, const float *x, float *out,
                              int64_t offset, size_t row_bytes, unsigned int rows) {
#define LCR_MV_LAUNCH(kernel) \
    kernel<<<rows, LCR_BLOCK_SIZE, 0, stream>>>(blob, in, x, out, offset, row_bytes)
    switch (type) {
    case LCR_F32:  LCR_MV_LAUNCH(lcr_matvec_f32);  break;
    case LCR_F16:  LCR_MV_LAUNCH(lcr_matvec_f16);  break;
    case LCR_Q8_0: LCR_MV_LAUNCH(lcr_matvec_q8_0); break;
    case LCR_Q4_K: LCR_MV_LAUNCH(lcr_matvec_q4_k); break;
    case LCR_Q5_K: LCR_MV_LAUNCH(lcr_matvec_q5_k); break;
    case LCR_Q6_K: LCR_MV_LAUNCH(lcr_matvec_q6_k); break;
    case LCR_BF16: LCR_MV_LAUNCH(lcr_matvec_bf16); break;
    default: break;
    }
#undef LCR_MV_LAUNCH
}

__global__ void lcr_row_kernel(const uint8_t *blob, int type, int in, float *out,
                               int64_t offset, size_t row_bytes) {
    const uint8_t *w = blob + offset;
    for (int n = threadIdx.x; n < in; n += LCR_BLOCK_SIZE) {
        out[n] = lcr_dequant_at(type, w, n);
    }
}

struct LcrContext {
    int device;
    char name[256];
    cudaStream_t stream;
    void *d_blob;
    float *d_x;
    float *d_y;
    LcrWeightMeta *metas;
    int count;
};

// GGML row size: bytes = in / block_size * block_bytes.
static int lcr_row_bytes(int type, int64_t in, size_t *out) {
    int bs;
    size_t bb;
    switch (type) {
    case LCR_F32:  bs = 1;  bb = 4;   break;
    case LCR_F16:  bs = 1;  bb = 2;   break;
    case LCR_BF16: bs = 1;  bb = 2;   break;
    case LCR_Q8_0: bs = 32; bb = 34;  break;
    case LCR_Q4_K: bs = LCR_QK_K; bb = sizeof(LcrBlockQ4_K); break;
    case LCR_Q5_K: bs = LCR_QK_K; bb = sizeof(LcrBlockQ5_K); break;
    case LCR_Q6_K: bs = LCR_QK_K; bb = sizeof(LcrBlockQ6_K); break;
    default: return LCR_ERR_TYPE;
    }
    if (in <= 0 || in % bs != 0) {
        return LCR_ERR_ARG;
    }
    *out = (size_t)(in / bs) * bb;
    return LCR_OK;
}

static void lcr_release_device(LcrContext *c) {
    if (c->d_blob) { cudaFree(c->d_blob); c->d_blob = NULL; }
    if (c->d_x)    { cudaFree(c->d_x);    c->d_x = NULL; }
    if (c->d_y)    { cudaFree(c->d_y);    c->d_y = NULL; }
    free(c->metas);
    c->metas = NULL;
    c->count = 0;
}

int lcr_gpu_init(void **ctx_out, int device) {
    if (!ctx_out) {
        return LCR_ERR_ARG;
    }
    *ctx_out = NULL;

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count) {
        return LCR_ERR_CUDA;
    }
    if (cudaSetDevice(device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }

    LcrContext *c = (LcrContext *)calloc(1, sizeof(LcrContext));
    if (!c) {
        return LCR_ERR_MEM;
    }
    c->device = device;
    snprintf(c->name, sizeof(c->name), "%s (sm_%d%d, CUDA %d.%d)",
             prop.name, prop.major, prop.minor, CUDART_VERSION / 1000,
             (CUDART_VERSION / 10) % 100);
    if (cudaStreamCreate(&c->stream) != cudaSuccess) {
        free(c);
        return LCR_ERR_CUDA;
    }
    *ctx_out = c;
    return LCR_OK;
}

void lcr_gpu_free(void *ctx) {
    if (!ctx) {
        return;
    }
    LcrContext *c = (LcrContext *)ctx;
    cudaSetDevice(c->device);
    cudaStreamSynchronize(c->stream);
    cudaStreamDestroy(c->stream);
    lcr_release_device(c);
    free(c);
}

int lcr_gpu_upload(void *ctx, const void *blob, size_t bytes,
                   const LcrWeightMeta *metas, int count) {
    if (!ctx || !blob || !metas || count <= 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }

    size_t max_in = 0, max_dim = 0;
    for (int i = 0; i < count; ++i) {
        const LcrWeightMeta *m = &metas[i];
        size_t rb;
        int rc = lcr_row_bytes(m->type, m->in, &rb);
        if (rc != LCR_OK || m->out <= 0 || m->offset < 0) {
            return rc != LCR_OK ? rc : LCR_ERR_ARG;
        }
        if ((uint64_t)m->offset > (uint64_t)bytes ||
            (uint64_t)m->out > (uint64_t)bytes / rb ||
            (uint64_t)m->offset > (uint64_t)bytes - (uint64_t)m->out * rb) {
            return LCR_ERR_ARG;
        }
        if ((uint64_t)m->in > max_in) {
            max_in = (size_t)m->in;
        }
        if ((uint64_t)m->out > max_dim) {
            max_dim = (size_t)m->out;
        }
    }
    if (max_dim > max_in) {
        max_in = max_dim;
    }

    LcrWeightMeta *host_metas =
        (LcrWeightMeta *)malloc((size_t)count * sizeof(LcrWeightMeta));
    if (!host_metas) {
        return LCR_ERR_MEM;
    }
    memcpy(host_metas, metas, (size_t)count * sizeof(LcrWeightMeta));

    void *d_blob = NULL;
    float *d_x = NULL;
    float *d_y = NULL;
    if (cudaMalloc(&d_blob, bytes) != cudaSuccess ||
        cudaMalloc(&d_x, max_in * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_y, max_in * sizeof(float)) != cudaSuccess) {
        if (d_blob) cudaFree(d_blob);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        free(host_metas);
        return LCR_ERR_CUDA;
    }
    if (cudaMemcpyAsync(d_blob, blob, bytes, cudaMemcpyHostToDevice, c->stream) != cudaSuccess) {
        cudaFree(d_blob);
        cudaFree(d_x);
        cudaFree(d_y);
        free(host_metas);
        return LCR_ERR_CUDA;
    }

    lcr_release_device(c);
    c->d_blob = d_blob;
    c->d_x = d_x;
    c->d_y = d_y;
    c->metas = host_metas;
    c->count = count;
    return LCR_OK;
}

int lcr_gpu_matvec(void *ctx, int index, const float *x, float *out) {
    if (!ctx || !x || !out) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (index < 0 || index >= c->count || !c->d_blob) {
        return LCR_ERR_RANGE;
    }
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }

    const LcrWeightMeta *m = &c->metas[index];
    size_t rb;
    int rc = lcr_row_bytes(m->type, m->in, &rb);
    if (rc != LCR_OK) {
        return rc;
    }
    size_t nf = (size_t)m->in;
    size_t no = (size_t)m->out;

    if (cudaMemcpyAsync(c->d_x, x, nf * sizeof(float), cudaMemcpyHostToDevice,
                        c->stream) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    lcr_launch_matvec(c->stream, (const uint8_t *)c->d_blob, m->type, (int)m->in,
                      c->d_x, c->d_y, m->offset, rb, (unsigned int)no);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    if (cudaMemcpyAsync(out, c->d_y, no * sizeof(float), cudaMemcpyDeviceToHost,
                        c->stream) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    if (cudaStreamSynchronize(c->stream) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_gpu_row(void *ctx, int index, int row, float *out) {
    if (!ctx || !out) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (index < 0 || index >= c->count || !c->d_blob) {
        return LCR_ERR_RANGE;
    }
    const LcrWeightMeta *m = &c->metas[index];
    if (row < 0 || row >= m->out) {
        return LCR_ERR_RANGE;
    }
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }

    size_t rb;
    int rc = lcr_row_bytes(m->type, m->in, &rb);
    if (rc != LCR_OK) {
        return rc;
    }
    size_t nf = (size_t)m->in;

    lcr_row_kernel<<<1, LCR_BLOCK_SIZE, 0, c->stream>>>(
        (const uint8_t *)c->d_blob, m->type, (int)m->in, c->d_y,
        m->offset + (int64_t)row * (int64_t)rb, rb);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    if (cudaMemcpyAsync(out, c->d_y, nf * sizeof(float), cudaMemcpyDeviceToHost,
                        c->stream) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    if (cudaStreamSynchronize(c->stream) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_gpu_info(void *ctx, char *buf, int len) {
    if (!ctx || !buf || len <= 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    size_t n = strlen(c->name);
    if ((size_t)len <= n) {
        n = (size_t)len - 1;
    }
    memcpy(buf, c->name, n);
    buf[n] = '\0';
    return LCR_OK;
}

int lcr_gpu_sync(void *ctx) {
    if (!ctx) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return cudaStreamSynchronize(c->stream) == cudaSuccess ? LCR_OK : LCR_ERR_CUDA;
}

// ---------------------------------------------------------------------------
// Per-token Qwen3.5 graph operations.
//
// Every pointer argument of the lcr_op_* entry points below is a DEVICE
// pointer. This keeps the whole forward graph resident on the GPU. The
// host-facing lcr_gpu_matvec/lcr_gpu_row above are unchanged.
// Each entry point sets the context device, launches on the context stream and
// returns immediately; the caller issues one lcr_gpu_sync per token. Work on a
// single stream is ordered, so later ops see earlier results without syncs.
// ---------------------------------------------------------------------------

// Block reductions over LCR_BLOCK_SIZE threads. tmp has one slot per warp.
__device__ __forceinline__ float lcr_block_sum(float v, float *tmp) {
    for (int s = 16; s > 0; s >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, s);
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        tmp[wid] = v;
    }
    __syncthreads();
    float r = threadIdx.x < (LCR_BLOCK_SIZE / 32) ? tmp[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        for (int s = (LCR_BLOCK_SIZE / 32) / 2; s > 0; s >>= 1) {
            r += __shfl_down_sync(0xFFFFFFFFu, r, s);
        }
        if (threadIdx.x == 0) {
            tmp[0] = r;
        }
    }
    __syncthreads();
    return tmp[0];
}

__device__ __forceinline__ float lcr_block_max(float v, float *tmp) {
    for (int s = 16; s > 0; s >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFu, v, s));
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        tmp[wid] = v;
    }
    __syncthreads();
    float r = threadIdx.x < (LCR_BLOCK_SIZE / 32) ? tmp[threadIdx.x] : -INFINITY;
    if (threadIdx.x < 32) {
        for (int s = (LCR_BLOCK_SIZE / 32) / 2; s > 0; s >>= 1) {
            r = fmaxf(r, __shfl_down_sync(0xFFFFFFFFu, r, s));
        }
        if (threadIdx.x == 0) {
            tmp[0] = r;
        }
    }
    __syncthreads();
    return tmp[0];
}

__global__ void lcr_rmsnorm_kernel(const float *x, const float *w, float *y,
                                   int n, float eps) {
    __shared__ float tmp[LCR_BLOCK_SIZE / 32];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += LCR_BLOCK_SIZE) {
        ss += x[i] * x[i];
    }
    const float inv = rsqrtf(lcr_block_sum(ss, tmp) / (float)n + eps);
    for (int i = tid; i < n; i += LCR_BLOCK_SIZE) {
        y[i] = x[i] * w[i] * inv;
    }
}

__global__ void lcr_l2norm_kernel(float *x, int n, float eps) {
    __shared__ float tmp[LCR_BLOCK_SIZE / 32];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += LCR_BLOCK_SIZE) {
        ss += x[i] * x[i];
    }
    const float inv = rsqrtf(lcr_block_sum(ss, tmp) + eps);
    for (int i = tid; i < n; i += LCR_BLOCK_SIZE) {
        x[i] *= inv;
    }
}

// NeoX rotary: rotate (x[i], x[i + n_rot/2]) by pos * base^(-2i/n_rot).
__global__ void lcr_rope_kernel(float *x, int n_rot, float base, int pos) {
    const int half = n_rot >> 1;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) {
        return;
    }
    const double ang =
        (double)pos * pow((double)base, -2.0 * (double)i / (double)n_rot);
    const float c = cosf((float)ang);
    const float s = sinf((float)ang);
    const float a = x[i];
    const float b = x[i + half];
    x[i] = a * c - b * s;
    x[i + half] = a * s + b * c;
}

__global__ void lcr_add_kernel(float *dst, const float *src, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        dst[i] += src[i];
    }
}

__global__ void lcr_mul_kernel(float *dst, const float *src, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        dst[i] *= src[i];
    }
}

__global__ void lcr_scale_kernel(float *x, int n, float scale) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        x[i] *= scale;
    }
}

__global__ void lcr_sigmoid_kernel(const float *x, float *y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

__global__ void lcr_gate_kernel(const float *alpha, const float *bias,
                                const float *a, float *gate, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const float v = alpha[i] + bias[i];
        const float softplus = fmaxf(v, 0.0f) + log1pf(expf(-fabsf(v)));
        gate[i] = a[i] * softplus;
    }
}

__global__ void lcr_silu_mul_kernel(const float *a, const float *b, float *y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const float v = a[i];
        y[i] = (v / (1.0f + expf(-v))) * b[i];
    }
}

// One thread per channel. Read the whole window before shifting hist, so
// aliasing qkv with out is harmless.
__global__ void lcr_conv_silu_kernel(const float *w, float *hist,
                                     const float *qkv, float *out,
                                     int channels, int k, int pitch) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    const float q = qkv[c];
    float sum = 0.0f;
    for (int j = 0; j < k - 1; ++j) {
        sum += w[(size_t)c * k + j] * hist[(size_t)j * pitch + c];
    }
    sum += w[(size_t)c * k + (k - 1)] * q;
    out[c] = sum / (1.0f + expf(-sum));
    for (int j = 0; j < k - 2; ++j) {
        hist[(size_t)j * pitch + c] = hist[(size_t)(j + 1) * pitch + c];
    }
    if (k >= 2) {
        hist[(size_t)(k - 2) * pitch + c] = q;
    }
}

// One block per value head. Shared k/q and scalar beta/decay; each thread owns
// whole rows j of state, so no cross-thread state synchronization is needed.
__global__ void lcr_gdn_kernel(const float *q, const float *k, const float *v,
                               const float *gate, const float *beta, float *state,
                               float *out, int n_k, int d) {
    extern __shared__ float smem[];
    const int h = blockIdx.x;
    const int kh = h % n_k;
    const int tid = threadIdx.x;
    float *sk = smem;
    float *sq = smem + d;
    for (int t = tid; t < d; t += blockDim.x) {
        sk[t] = k[(size_t)kh * d + t];
        sq[t] = q[(size_t)kh * d + t];
    }
    if (tid == 0) {
        smem[2 * d] = expf(gate[h]);
        smem[2 * d + 1] = beta[h];
    }
    __syncthreads();
    const float decay = smem[2 * d];
    const float b = smem[2 * d + 1];
    const float inv = rsqrtf((float)d);
    for (int j = tid; j < d; j += blockDim.x) {
        float *row = state + (size_t)h * d * d + (size_t)j * d;
        float kdot = 0.0f;
        for (int t = 0; t < d; ++t) {
            row[t] *= decay;
            kdot += row[t] * sk[t];
        }
        const float delta = (v[(size_t)h * d + j] - kdot) * b;
        float acc = 0.0f;
        for (int t = 0; t < d; ++t) {
            row[t] += delta * sk[t];
            acc += row[t] * sq[t];
        }
        out[(size_t)h * d + j] = acc * inv;
    }
}

__global__ void lcr_attn_store_kernel(const float *k, const float *v,
                                      float *kc, float *vc,
                                      int n_kv, int seq_cap, int seq_pos, int d) {
    const int kv = blockIdx.x;
    for (int t = threadIdx.x; t < d; t += LCR_BLOCK_SIZE) {
        const size_t off = ((size_t)kv * seq_cap + seq_pos) * d + t;
        kc[off] = k[(size_t)kv * d + t];
        vc[off] = v[(size_t)kv * d + t];
    }
}

// Streaming (online) softmax: one dot product per cache slot, no seq_cap-sized
// shared array, so long contexts cannot exceed the shared memory limit.
__global__ void lcr_attn_kernel(const float *q, const float *g,
                                const float *kc, const float *vc, float *out,
                                int n_heads, int n_kv, int d,
                                int seq_pos, int seq_cap, float scale) {
    __shared__ float tmp[LCR_BLOCK_SIZE / 32];
    extern __shared__ float acc[];
    const int h = blockIdx.x;
    const int kv = h * n_kv / n_heads;
    const int tid = threadIdx.x;
    for (int j = tid; j < d; j += LCR_BLOCK_SIZE) {
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float s = 0.0f;
    for (int t = 0; t <= seq_pos; ++t) {
        const float *kt = kc + ((size_t)kv * seq_cap + t) * d;
        const float *vt = vc + ((size_t)kv * seq_cap + t) * d;
        float part = 0.0f;
        for (int j = tid; j < d; j += LCR_BLOCK_SIZE) {
            part += q[(size_t)h * d + j] * kt[j];
        }
        const float score = lcr_block_sum(part, tmp) * scale;
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);
        const float p = expf(score - m_new);
        for (int j = tid; j < d; j += LCR_BLOCK_SIZE) {
            acc[j] = acc[j] * alpha + p * vt[j];
        }
        s = s * alpha + p;
        m = m_new;
    }
    const float inv = 1.0f / s;
    for (int j = tid; j < d; j += LCR_BLOCK_SIZE) {
        const float gate = 1.0f / (1.0f + expf(-g[(size_t)h * d + j]));
        out[(size_t)h * d + j] = acc[j] * inv * gate;
    }
}

__global__ void lcr_argmax_kernel(const float *x, int n, int *out) {
    __shared__ float sv[LCR_BLOCK_SIZE / 32];
    __shared__ int si[LCR_BLOCK_SIZE / 32];
    const int tid = threadIdx.x;
    float bv = -INFINITY;
    int bi = 0;
    for (int i = tid; i < n; i += LCR_BLOCK_SIZE) {
        if (x[i] > bv) {
            bv = x[i];
            bi = i;
        }
    }
    for (int s = 16; s > 0; s >>= 1) {
        const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, s);
        const int oi = __shfl_down_sync(0xFFFFFFFFu, bi, s);
        if (ov > bv) {
            bv = ov;
            bi = oi;
        }
    }
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (lane == 0) {
        sv[wid] = bv;
        si[wid] = bi;
    }
    __syncthreads();
    if (tid == 0) {
        float v = -INFINITY;
        int idx = 0;
        for (int w = 0; w < LCR_BLOCK_SIZE / 32; ++w) {
            if (sv[w] > v) {
                v = sv[w];
                idx = si[w];
            }
        }
        out[0] = idx;
    }
}

// --- fused per-head kernels -------------------------------------------------

// qg[h] holds q[head_dim] followed by gate[head_dim].
__global__ void lcr_split_qg_kernel(const float *qg, float *q, float *g,
                                    int head_dim) {
    const int h = blockIdx.x;
    const size_t base = (size_t)h * 2 * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += LCR_BLOCK_SIZE) {
        q[(size_t)h * head_dim + i] = qg[base + i];
        g[(size_t)h * head_dim + i] = qg[base + head_dim + i];
    }
}

// RMSNorm has no eps argument in the ABI; Qwen3.5 uses 1e-6.
#define LCR_QK_NORM_EPS 1e-6f

__global__ void lcr_qk_norm_rope_kernel(float *x, const float *w, int head_dim,
                                        int n_rot, float base, int pos) {
    __shared__ float tmp[LCR_BLOCK_SIZE / 32];
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    float *xh = x + (size_t)h * head_dim;

    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += LCR_BLOCK_SIZE) {
        ss += xh[i] * xh[i];
    }
    const float inv = rsqrtf(lcr_block_sum(ss, tmp) / (float)head_dim + LCR_QK_NORM_EPS);
    for (int i = tid; i < head_dim; i += LCR_BLOCK_SIZE) {
        xh[i] = xh[i] * w[i] * inv;
    }
    __syncthreads();

    const int half = n_rot >> 1;
    for (int i = tid; i < half; i += LCR_BLOCK_SIZE) {
        const double ang =
            (double)pos * pow((double)base, -2.0 * (double)i / (double)n_rot);
        const float c = cosf((float)ang);
        const float s = sinf((float)ang);
        const float a = xh[i];
        const float b = xh[i + half];
        xh[i] = a * c - b * s;
        xh[i + half] = a * s + b * c;
    }
}

__global__ void lcr_l2norm_heads_kernel(float *x, int head_dim, float eps) {
    __shared__ float tmp[LCR_BLOCK_SIZE / 32];
    const int tid = threadIdx.x;
    float *xh = x + (size_t)blockIdx.x * head_dim;
    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += LCR_BLOCK_SIZE) {
        ss += xh[i] * xh[i];
    }
    const float inv = rsqrtf(lcr_block_sum(ss, tmp) + eps);
    for (int i = tid; i < head_dim; i += LCR_BLOCK_SIZE) {
        xh[i] *= inv;
    }
}

__global__ void lcr_gdn_post_kernel(const float *gdn, const float *z,
                                    const float *norm_w, float *y,
                                    int head_dim, float eps) {
    __shared__ float tmp[LCR_BLOCK_SIZE / 32];
    const int tid = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * head_dim;
    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += LCR_BLOCK_SIZE) {
        ss += gdn[base + i] * gdn[base + i];
    }
    const float inv = rsqrtf(lcr_block_sum(ss, tmp) / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += LCR_BLOCK_SIZE) {
        const float v = z[base + i];
        const float silu = v / (1.0f + expf(-v));
        y[base + i] = silu * (gdn[base + i] * norm_w[i] * inv);
    }
}

// --- device-resident helpers ------------------------------------------------

int lcr_gpu_zero(void *ctx, void *ptr, size_t bytes) {
    if (!ctx || !ptr) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    if (cudaMemsetAsync(ptr, 0, bytes, c->stream) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_gpu_row_device(void *ctx, int index, int row, float *dst) {
    if (!ctx || !dst) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (index < 0 || index >= c->count || !c->d_blob) {
        return LCR_ERR_RANGE;
    }
    const LcrWeightMeta *m = &c->metas[index];
    if (row < 0 || row >= m->out) {
        return LCR_ERR_RANGE;
    }
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    size_t rb;
    int rc = lcr_row_bytes(m->type, m->in, &rb);
    if (rc != LCR_OK) {
        return rc;
    }
    lcr_row_kernel<<<1, LCR_BLOCK_SIZE, 0, c->stream>>>(
        (const uint8_t *)c->d_blob, m->type, (int)m->in, dst,
        m->offset + (int64_t)row * (int64_t)rb, rb);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_gpu_matvec_device(void *ctx, int index, const float *x, float *out) {
    if (!ctx || !x || !out) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (index < 0 || index >= c->count || !c->d_blob) {
        return LCR_ERR_RANGE;
    }
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    const LcrWeightMeta *m = &c->metas[index];
    size_t rb;
    int rc = lcr_row_bytes(m->type, m->in, &rb);
    if (rc != LCR_OK) {
        return rc;
    }
    size_t no = (size_t)m->out;
    lcr_launch_matvec(c->stream, (const uint8_t *)c->d_blob, m->type, (int)m->in,
                      x, out, m->offset, rb, (unsigned int)no);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_gpu_read_i32(void *ctx, const void *src, int *dst) {
    if (!ctx || !src || !dst) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    if (cudaMemcpyAsync(dst, src, sizeof(int), cudaMemcpyDeviceToHost,
                        c->stream) != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_gpu_buf(void *ctx, size_t bytes, void **out) {
    if (!ctx || !out || bytes == 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    *out = NULL;
    return cudaMalloc(out, bytes) == cudaSuccess ? LCR_OK : LCR_ERR_CUDA;
}

void lcr_gpu_buf_free(void *ctx, void *ptr) {
    if (!ctx) {
        return;
    }
    LcrContext *c = (LcrContext *)ctx;
    cudaSetDevice(c->device);
    if (ptr) {
        cudaFree(ptr);
    }
}

void *lcr_gpu_weight_ptr(void *ctx, int index) {
    if (!ctx) {
        return NULL;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (index < 0 || index >= c->count || !c->d_blob) {
        return NULL;
    }
    return (uint8_t *)c->d_blob + c->metas[index].offset;
}

// --- per-token ops ----------------------------------------------------------

int lcr_op_rmsnorm(void *ctx, const float *x, const float *w, float *y,
                   int n, float eps) {
    if (!ctx || !x || !w || !y || n <= 0 || eps < 0.0f) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    lcr_rmsnorm_kernel<<<1, LCR_BLOCK_SIZE, 0, c->stream>>>(x, w, y, n, eps);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_l2norm(void *ctx, float *x, int n, float eps) {
    if (!ctx || !x || n <= 0 || eps < 0.0f) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    lcr_l2norm_kernel<<<1, LCR_BLOCK_SIZE, 0, c->stream>>>(x, n, eps);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_rope(void *ctx, float *x, int n, int n_rot, float base, int pos) {
    if (!ctx || !x || n <= 0 || n_rot <= 0 || n_rot > n ||
        (n_rot & 1) || !(base > 0.0f) || pos < 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    const int pairs = n_rot >> 1;
    const int blocks = (pairs + LCR_BLOCK_SIZE - 1) / LCR_BLOCK_SIZE;
    lcr_rope_kernel<<<blocks, LCR_BLOCK_SIZE, 0, c->stream>>>(x, n_rot, base, pos);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_add(void *ctx, float *dst, const float *src, int n) {
    if (!ctx || !dst || !src || n <= 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    const int blocks = (n + LCR_BLOCK_SIZE - 1) / LCR_BLOCK_SIZE;
    lcr_add_kernel<<<blocks, LCR_BLOCK_SIZE, 0, c->stream>>>(dst, src, n);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_mul(void *ctx, float *dst, const float *src, int n) {
    if (!ctx || !dst || !src || n <= 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    const int blocks = (n + LCR_BLOCK_SIZE - 1) / LCR_BLOCK_SIZE;
    lcr_mul_kernel<<<blocks, LCR_BLOCK_SIZE, 0, c->stream>>>(dst, src, n);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_scale(void *ctx, float *x, int n, float scale) {
    if (!ctx || !x || n <= 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    const int blocks = (n + LCR_BLOCK_SIZE - 1) / LCR_BLOCK_SIZE;
    lcr_scale_kernel<<<blocks, LCR_BLOCK_SIZE, 0, c->stream>>>(x, n, scale);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_sigmoid(void *ctx, const float *x, float *y, int n) {
    if (!ctx || !x || !y || n <= 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    const int blocks = (n + LCR_BLOCK_SIZE - 1) / LCR_BLOCK_SIZE;
    lcr_sigmoid_kernel<<<blocks, LCR_BLOCK_SIZE, 0, c->stream>>>(x, y, n);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_gate(void *ctx, const float *alpha, const float *bias, const float *a,
                float *gate, int n) {
    if (!ctx || !alpha || !bias || !a || !gate || n <= 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    const int blocks = (n + LCR_BLOCK_SIZE - 1) / LCR_BLOCK_SIZE;
    lcr_gate_kernel<<<blocks, LCR_BLOCK_SIZE, 0, c->stream>>>(alpha, bias, a, gate, n);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_silu_mul(void *ctx, const float *a, const float *b, float *y, int n) {
    if (!ctx || !a || !b || !y || n <= 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    const int blocks = (n + LCR_BLOCK_SIZE - 1) / LCR_BLOCK_SIZE;
    lcr_silu_mul_kernel<<<blocks, LCR_BLOCK_SIZE, 0, c->stream>>>(a, b, y, n);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_conv_silu(void *ctx, const float *w, const float *hist, const float *qkv,
                     float *out, int channels, int k, int stride) {
    if (!ctx || !w || !qkv || !out || channels <= 0 || k < 1 || stride < 0 ||
        (k > 1 && !hist)) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    // shortcut: stride is used as an optional padded row pitch; the documented
    // contiguous layout is recovered when stride < channels (upgrade if conv
    // gains a temporal stride > 1).
    const int pitch = stride >= channels ? stride : channels;
    const int blocks = (channels + LCR_BLOCK_SIZE - 1) / LCR_BLOCK_SIZE;
    // hist is const in the public ABI but the op updates it in place.
    lcr_conv_silu_kernel<<<blocks, LCR_BLOCK_SIZE, 0, c->stream>>>(
        w, (float *)hist, qkv, out, channels, k, pitch);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_gdn(void *ctx, const float *q, const float *k, const float *v,
               const float *gate, const float *beta, float *state, float *out,
               int n_v, int n_k, int d) {
    if (!ctx || !q || !k || !v || !gate || !beta || !state || !out ||
        n_v <= 0 || n_k <= 0 || d <= 0 || n_v % n_k != 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    const size_t smem_bytes = (size_t)(2 * d + 2) * sizeof(float);
    lcr_gdn_kernel<<<n_v, LCR_BLOCK_SIZE, smem_bytes, c->stream>>>(
        q, k, v, gate, beta, state, out, n_k, d);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_attn(void *ctx, const float *q, const float *k, const float *v,
                const float *g, float *kc, float *vc, float *out,
                int n_heads, int n_kv, int d, int seq_pos, int seq_cap,
                float scale) {
    if (!ctx || !q || !k || !v || !g || !kc || !vc || !out ||
        n_heads <= 0 || n_kv <= 0 || d <= 0 || n_heads % n_kv != 0 ||
        seq_cap <= 0 || seq_pos < 0 || seq_pos >= seq_cap) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    lcr_attn_store_kernel<<<n_kv, LCR_BLOCK_SIZE, 0, c->stream>>>(
        k, v, kc, vc, n_kv, seq_cap, seq_pos, d);
    // Same stream: the cache write completes before any query head reads it.
    lcr_attn_kernel<<<n_heads, LCR_BLOCK_SIZE, (size_t)d * sizeof(float), c->stream>>>(
        q, g, kc, vc, out, n_heads, n_kv, d, seq_pos, seq_cap, scale);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_argmax(void *ctx, const float *x, int n, int *out) {
    if (!ctx || !x || !out || n <= 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    lcr_argmax_kernel<<<1, LCR_BLOCK_SIZE, 0, c->stream>>>(x, n, out);
    if (cudaGetLastError() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    return LCR_OK;
}

int lcr_op_split_qg(void *ctx, const float *qg, float *q, float *g,
                    int n_heads, int head_dim) {
    if (!ctx || !qg || !q || !g || n_heads <= 0 || head_dim <= 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    lcr_split_qg_kernel<<<n_heads, LCR_BLOCK_SIZE, 0, c->stream>>>(
        qg, q, g, head_dim);
    return cudaGetLastError() == cudaSuccess ? LCR_OK : LCR_ERR_CUDA;
}

int lcr_op_qk_norm_rope(void *ctx, float *x, const float *w,
                        int n_heads, int head_dim, int n_rot, float base, int pos) {
    if (!ctx || !x || !w || n_heads <= 0 || head_dim <= 0 ||
        n_rot <= 0 || n_rot > head_dim || (n_rot & 1) || !(base > 0.0f) || pos < 0) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    lcr_qk_norm_rope_kernel<<<n_heads, LCR_BLOCK_SIZE, 0, c->stream>>>(
        x, w, head_dim, n_rot, base, pos);
    return cudaGetLastError() == cudaSuccess ? LCR_OK : LCR_ERR_CUDA;
}

int lcr_op_l2norm_heads(void *ctx, float *x, int n_heads, int head_dim, float eps) {
    if (!ctx || !x || n_heads <= 0 || head_dim <= 0 || eps < 0.0f) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    lcr_l2norm_heads_kernel<<<n_heads, LCR_BLOCK_SIZE, 0, c->stream>>>(
        x, head_dim, eps);
    return cudaGetLastError() == cudaSuccess ? LCR_OK : LCR_ERR_CUDA;
}

int lcr_op_gdn_post(void *ctx, const float *gdn, const float *z, const float *norm_w,
                    float *y, int n_heads, int head_dim, float eps) {
    if (!ctx || !gdn || !z || !norm_w || !y || n_heads <= 0 || head_dim <= 0 ||
        eps < 0.0f) {
        return LCR_ERR_ARG;
    }
    LcrContext *c = (LcrContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    lcr_gdn_post_kernel<<<n_heads, LCR_BLOCK_SIZE, 0, c->stream>>>(
        gdn, z, norm_w, y, head_dim, eps);
    return cudaGetLastError() == cudaSuccess ? LCR_OK : LCR_ERR_CUDA;
}
