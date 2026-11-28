// Batched dequant matmul for prompt prefill. Shares the context and stream with
// lcr_kernels.cu but launches one block per (output row, token) so a whole chunk
// of prompt tokens is processed in a single kernel launch.

#include "lcr_kernels.cuh"
#include "lcr_batch.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdlib.h>
#include <string.h>

enum {
    BATCH_OK        = 0,
    BATCH_ERR_ARG   = -1,
    BATCH_ERR_CUDA  = -2,
    BATCH_ERR_TYPE  = -3,
    BATCH_ERR_RANGE = -4
};

enum {
    BATCH_F32  = 0,
    BATCH_F16  = 1,
    BATCH_Q8_0 = 8,
    BATCH_Q4_K = 12,
    BATCH_Q5_K = 13,
    BATCH_Q6_K = 14,
    BATCH_BF16 = 30
};

#define BATCH_QK_K 256
#define BATCH_K_SCALE_SIZE 12
#define BATCH_BLOCK_SIZE 256

struct BatchBlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BatchBlockQ8_0) == 34, "q8_0 block size");

struct BatchBlockQ4_K {
    uint32_t dm;
    uint8_t scales[BATCH_K_SCALE_SIZE];
    uint8_t qs[BATCH_QK_K / 2];
};
static_assert(sizeof(BatchBlockQ4_K) == 144, "q4_K block size");

struct BatchBlockQ5_K {
    uint32_t dm;
    uint8_t scales[BATCH_K_SCALE_SIZE];
    uint8_t qh[BATCH_QK_K / 8];
    uint8_t qs[BATCH_QK_K / 2];
};
static_assert(sizeof(BatchBlockQ5_K) == 176, "q5_K block size");

struct BatchBlockQ6_K {
    uint8_t ql[BATCH_QK_K / 2];
    uint8_t qh[BATCH_QK_K / 4];
    int8_t scales[BATCH_QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(BatchBlockQ6_K) == 210, "q6_K block size");

// Dequant helpers copied from lcr_kernels.cu. Kept private so matvec tuning in
// that file cannot silently change prefill numerics.
__device__ __forceinline__ float batch_half2float(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ float batch_bf162float(uint16_t b) {
    return __bfloat162float(__ushort_as_bfloat16(b));
}

__device__ __forceinline__ void batch_get_scale_min_k4(int j, const uint8_t *q,
                                                       uint8_t &d, uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// One warp per (output row, token) pair; a CTA of 256 threads covers eight
// pairs. Mirrors the matvec kernels in lcr_kernels.cu: each warp walks whole
// quant blocks, reads group scales once per subgroup, and reduces with shuffles
// only. All loops are type-specialized; no switch remains per element.
__device__ __forceinline__ void batch_mv_reduce(float sum, float *out) {
    for (int s = 16; s > 0; s >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, s);
    }
    if ((threadIdx.x & 31) == 0) {
        *out = sum;
    }
}

__device__ __forceinline__ bool batch_vec_ok(const void *w, const void *x, int align) {
    return ((uintptr_t)w & (align - 1)) == 0 &&
           ((uintptr_t)x & (align - 1)) == 0;
}

static int batch_row_bytes(int type, int64_t in, size_t *out) {
    int bs;
    size_t bb;
    switch (type) {
    case BATCH_F32:  bs = 1;  bb = 4;  break;
    case BATCH_F16:  bs = 1;  bb = 2;  break;
    case BATCH_BF16: bs = 1;  bb = 2;  break;
    case BATCH_Q8_0: bs = 32; bb = 34; break;
    case BATCH_Q4_K: bs = BATCH_QK_K; bb = sizeof(BatchBlockQ4_K); break;
    case BATCH_Q5_K: bs = BATCH_QK_K; bb = sizeof(BatchBlockQ5_K); break;
    case BATCH_Q6_K: bs = BATCH_QK_K; bb = sizeof(BatchBlockQ6_K); break;
    default: return BATCH_ERR_TYPE;
    }
    if (in <= 0 || in % bs != 0) {
        return BATCH_ERR_ARG;
    }
    *out = (size_t)(in / bs) * bb;
    return BATCH_OK;
}

// Exact mirror of LcrContext in lcr_kernels.cu. Both translation units are
// built by the same nvcc invocation. Update together if that struct changes.
struct BatchContext {
    int device;
    char name[256];
    cudaStream_t stream;
    void *d_blob;
    float *d_x;
    float *d_y;
    LcrWeightMeta *metas;
    int count;
};

__global__ void lcr_batch_matmul_f32(const uint8_t * __restrict__ blob, int in,
                                     const float * __restrict__ x,
                                     float * __restrict__ out, int64_t offset,
                                     size_t row_bytes, int out_rows,
                                     int x_stride, int out_stride) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int row = blockIdx.x * (BATCH_BLOCK_SIZE / 32) + wid;
    const int tok = blockIdx.y;
    if (row >= out_rows) return;
    const float * __restrict__ w =
        (const float *)(blob + offset + (size_t)row * row_bytes);
    const float * __restrict__ xr = x + (size_t)tok * x_stride;
    float sum = 0.0f;
    if (((in & 3) == 0) && batch_vec_ok(w, xr, 16)) {
        for (int n = lane * 4; n + 4 <= in; n += 32 * 4) {
            const float4 wv = __ldg((const float4 *)(w + n));
            const float4 xv = __ldg((const float4 *)(xr + n));
            sum += (wv.x * xv.x + wv.y * xv.y) + (wv.z * xv.z + wv.w * xv.w);
        }
    } else {
        for (int n = lane; n < in; n += 32) {
            sum += __ldg(w + n) * __ldg(xr + n);
        }
    }
    batch_mv_reduce(sum, out + (size_t)tok * out_stride + row);
}

__global__ void lcr_batch_matmul_f16(const uint8_t * __restrict__ blob, int in,
                                     const float * __restrict__ x,
                                     float * __restrict__ out, int64_t offset,
                                     size_t row_bytes, int out_rows,
                                     int x_stride, int out_stride) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int row = blockIdx.x * (BATCH_BLOCK_SIZE / 32) + wid;
    const int tok = blockIdx.y;
    if (row >= out_rows) return;
    const uint16_t * __restrict__ w =
        (const uint16_t *)(blob + offset + (size_t)row * row_bytes);
    const float * __restrict__ xr = x + (size_t)tok * x_stride;
    float sum = 0.0f;
    if (((in & 3) == 0) && batch_vec_ok(w, xr, 8)) {
        for (int n = lane * 4; n + 4 <= in; n += 32 * 4) {
            const uint2 wv = __ldg((const uint2 *)(w + n));
            const float4 xv = __ldg((const float4 *)(xr + n));
            sum += batch_half2float((uint16_t)wv.x) * xv.x;
            sum += batch_half2float((uint16_t)(wv.x >> 16)) * xv.y;
            sum += batch_half2float((uint16_t)wv.y) * xv.z;
            sum += batch_half2float((uint16_t)(wv.y >> 16)) * xv.w;
        }
    } else {
        for (int n = lane; n < in; n += 32) {
            sum += batch_half2float(__ldg(w + n)) * __ldg(xr + n);
        }
    }
    batch_mv_reduce(sum, out + (size_t)tok * out_stride + row);
}

__global__ void lcr_batch_matmul_bf16(const uint8_t * __restrict__ blob, int in,
                                      const float * __restrict__ x,
                                      float * __restrict__ out, int64_t offset,
                                      size_t row_bytes, int out_rows,
                                      int x_stride, int out_stride) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int row = blockIdx.x * (BATCH_BLOCK_SIZE / 32) + wid;
    const int tok = blockIdx.y;
    if (row >= out_rows) return;
    const uint16_t * __restrict__ w =
        (const uint16_t *)(blob + offset + (size_t)row * row_bytes);
    const float * __restrict__ xr = x + (size_t)tok * x_stride;
    float sum = 0.0f;
    if (((in & 3) == 0) && batch_vec_ok(w, xr, 8)) {
        for (int n = lane * 4; n + 4 <= in; n += 32 * 4) {
            const uint2 wv = __ldg((const uint2 *)(w + n));
            const float4 xv = __ldg((const float4 *)(xr + n));
            sum += batch_bf162float((uint16_t)wv.x) * xv.x;
            sum += batch_bf162float((uint16_t)(wv.x >> 16)) * xv.y;
            sum += batch_bf162float((uint16_t)wv.y) * xv.z;
            sum += batch_bf162float((uint16_t)(wv.y >> 16)) * xv.w;
        }
    } else {
        for (int n = lane; n < in; n += 32) {
            sum += batch_bf162float(__ldg(w + n)) * __ldg(xr + n);
        }
    }
    batch_mv_reduce(sum, out + (size_t)tok * out_stride + row);
}

__global__ void lcr_batch_matmul_q8_0(const uint8_t * __restrict__ blob, int in,
                                      const float * __restrict__ x,
                                      float * __restrict__ out, int64_t offset,
                                      size_t row_bytes, int out_rows,
                                      int x_stride, int out_stride) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int row = blockIdx.x * (BATCH_BLOCK_SIZE / 32) + wid;
    const int tok = blockIdx.y;
    if (row >= out_rows) return;
    const uint8_t * __restrict__ w = blob + offset + (size_t)row * row_bytes;
    const float * __restrict__ xr = x + (size_t)tok * x_stride;
    const int blocks = in >> 5;
    float sum = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const BatchBlockQ8_0 *blk =
            (const BatchBlockQ8_0 *)(w + (size_t)b * sizeof(BatchBlockQ8_0));
        float d = 0.0f;
        if (lane == 0) {
            d = batch_half2float(__ldg(&blk->d));
        }
        d = __shfl_sync(0xFFFFFFFFu, d, 0);
        const int q = (int)__ldg(&blk->qs[lane]);
        sum += ((float)q * d) * __ldg(xr + (size_t)b * 32 + lane);
    }
    batch_mv_reduce(sum, out + (size_t)tok * out_stride + row);
}

// Each lane walks the eight 32-element subgroups of every 256-element block,
// so each packed byte is read once and a scale/min pair is computed once per
// subgroup. Scales broadcast within each 8-lane subgroup via shuffles.
__global__ void lcr_batch_matmul_q4_k(const uint8_t * __restrict__ blob, int in,
                                      const float * __restrict__ x,
                                      float * __restrict__ out, int64_t offset,
                                      size_t row_bytes, int out_rows,
                                      int x_stride, int out_stride) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int row = blockIdx.x * (BATCH_BLOCK_SIZE / 32) + wid;
    const int tok = blockIdx.y;
    if (row >= out_rows) return;
    const uint8_t * __restrict__ w = blob + offset + (size_t)row * row_bytes;
    const float * __restrict__ xr = x + (size_t)tok * x_stride;
    const int blocks = in >> 8;
    const bool vx = (((uintptr_t)xr) & 15) == 0;
    float sum = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const BatchBlockQ4_K *blk =
            (const BatchBlockQ4_K *)(w + (size_t)b * sizeof(BatchBlockQ4_K));
        const uint32_t md = __ldg((const uint32_t *)&blk->dm);
        const float dall = batch_half2float((uint16_t)md);
        const float dmin = batch_half2float((uint16_t)(md >> 16));
        float acc = 0.0f;
        if (vx) {
            const int gbase = lane >> 3;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int g = gbase + t * 4;
                float d = 0.0f, dm = 0.0f;
                if ((lane & 7) == 0) {
                    uint8_t sc, m;
                    batch_get_scale_min_k4(g, blk->scales, sc, m);
                    d = dall * (float)sc;
                    dm = dmin * (float)m;
                }
                d = __shfl_sync(0xFFFFFFFFu, d, 0, 8);
                dm = __shfl_sync(0xFFFFFFFFu, dm, 0, 8);
                const uint32_t q = __ldg((const uint32_t *)
                    (blk->qs + 32 * (g >> 1) + 4 * (lane & 7)));
                const float4 xv = __ldg((const float4 *)
                    (xr + (size_t)b * BATCH_QK_K + g * 32 + 4 * (lane & 7)));
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
                    batch_get_scale_min_k4(s, blk->scales, sc, m);
                    d = dall * (float)sc;
                    dm = dmin * (float)m;
                }
                d = __shfl_sync(0xFFFFFFFFu, d, 0);
                dm = __shfl_sync(0xFFFFFFFFu, dm, 0);
                const uint8_t q = __ldg(&blk->qs[32 * (s >> 1) + lane]);
                const float vq = (s & 1) ? (float)(q >> 4) : (float)(q & 0xF);
                acc += (d * vq - dm) *
                       __ldg(xr + (size_t)b * BATCH_QK_K + s * 32 + lane);
            }
        }
        sum += acc;
    }
    batch_mv_reduce(sum, out + (size_t)tok * out_stride + row);
}

__global__ void lcr_batch_matmul_q5_k(const uint8_t * __restrict__ blob, int in,
                                      const float * __restrict__ x,
                                      float * __restrict__ out, int64_t offset,
                                      size_t row_bytes, int out_rows,
                                      int x_stride, int out_stride) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int row = blockIdx.x * (BATCH_BLOCK_SIZE / 32) + wid;
    const int tok = blockIdx.y;
    if (row >= out_rows) return;
    const uint8_t * __restrict__ w = blob + offset + (size_t)row * row_bytes;
    const float * __restrict__ xr = x + (size_t)tok * x_stride;
    const int blocks = in >> 8;
    const bool vx = (((uintptr_t)xr) & 15) == 0;
    float sum = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const BatchBlockQ5_K *blk =
            (const BatchBlockQ5_K *)(w + (size_t)b * sizeof(BatchBlockQ5_K));
        const uint32_t md = __ldg((const uint32_t *)&blk->dm);
        const float dall = batch_half2float((uint16_t)md);
        const float dmin = batch_half2float((uint16_t)(md >> 16));
        float acc = 0.0f;
        if (vx) {
            const int gbase = lane >> 3;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int g = gbase + t * 4;
                float d = 0.0f, dm = 0.0f;
                if ((lane & 7) == 0) {
                    uint8_t sc, m;
                    batch_get_scale_min_k4(g, blk->scales, sc, m);
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
                    (xr + (size_t)b * BATCH_QK_K + g * 32 + 4 * (lane & 7)));
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
                    batch_get_scale_min_k4(s, blk->scales, sc, m);
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
                       __ldg(xr + (size_t)b * BATCH_QK_K + s * 32 + lane);
            }
        }
        sum += acc;
    }
    batch_mv_reduce(sum, out + (size_t)tok * out_stride + row);
}

__global__ void lcr_batch_matmul_q6_k(const uint8_t * __restrict__ blob, int in,
                                      const float * __restrict__ x,
                                      float * __restrict__ out, int64_t offset,
                                      size_t row_bytes, int out_rows,
                                      int x_stride, int out_stride) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int row = blockIdx.x * (BATCH_BLOCK_SIZE / 32) + wid;
    const int tok = blockIdx.y;
    if (row >= out_rows) return;
    const uint8_t * __restrict__ w = blob + offset + (size_t)row * row_bytes;
    const float * __restrict__ xr = x + (size_t)tok * x_stride;
    const int blocks = in >> 8;
    const bool vx = (((uintptr_t)xr) & 15) == 0;
    float sum = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const BatchBlockQ6_K *blk =
            (const BatchBlockQ6_K *)(w + (size_t)b * sizeof(BatchBlockQ6_K));
        const float d = batch_half2float(__ldg(&blk->d));
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
                    (xr + (size_t)b * BATCH_QK_K + g * 32 + 4 * (lane & 7)));
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
                       __ldg(xr + (size_t)b * BATCH_QK_K + s * 32 + lane);
            }
        }
        sum += acc;
    }
    batch_mv_reduce(sum, out + (size_t)tok * out_stride + row);
}

// One switch per launch, never per element. Type is validated by caller.
static int batch_launch_matmul(cudaStream_t stream, const uint8_t *blob, int type,
                               int in, const float *x, float *out,
                               int64_t offset, size_t row_bytes, int out_rows,
                               int n_tokens, int x_stride, int out_stride) {
    (void)n_tokens;
#define BATCH_MV_LAUNCH(kernel) \
    kernel<<<dim3((out_rows + BATCH_BLOCK_SIZE / 32 - 1) / (BATCH_BLOCK_SIZE / 32), \
                  (unsigned int)n_tokens), BATCH_BLOCK_SIZE, 0, stream>>>( \
        blob, in, x, out, offset, row_bytes, out_rows, x_stride, out_stride)
    switch (type) {
    case BATCH_F32:  BATCH_MV_LAUNCH(lcr_batch_matmul_f32);  break;
    case BATCH_F16:  BATCH_MV_LAUNCH(lcr_batch_matmul_f16);  break;
    case BATCH_BF16: BATCH_MV_LAUNCH(lcr_batch_matmul_bf16); break;
    case BATCH_Q8_0: BATCH_MV_LAUNCH(lcr_batch_matmul_q8_0); break;
    case BATCH_Q4_K: BATCH_MV_LAUNCH(lcr_batch_matmul_q4_k); break;
    case BATCH_Q5_K: BATCH_MV_LAUNCH(lcr_batch_matmul_q5_k); break;
    case BATCH_Q6_K: BATCH_MV_LAUNCH(lcr_batch_matmul_q6_k); break;
    default: return BATCH_ERR_TYPE;
    }
#undef BATCH_MV_LAUNCH
    return BATCH_OK;
}

int lcr_batch_matmul(void *ctx, int index, const float *x, float *out,
                      int n_tokens, int x_stride, int out_stride) {
    if (!ctx || !x || !out || n_tokens <= 0 || x_stride <= 0 || out_stride <= 0) {
        return BATCH_ERR_ARG;
    }
    BatchContext *c = (BatchContext *)ctx;
    if (index < 0 || index >= c->count || !c->d_blob) {
        return BATCH_ERR_RANGE;
    }
    const LcrWeightMeta *m = &c->metas[index];
    size_t rb;
    int rc = batch_row_bytes(m->type, m->in, &rb);
    if (rc != BATCH_OK) {
        return rc;
    }
    if (m->out <= 0 || x_stride < m->in || out_stride < m->out) {
        return BATCH_ERR_ARG;
    }
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return BATCH_ERR_CUDA;
    }

    rc = batch_launch_matmul(c->stream, (const uint8_t *)c->d_blob, (int)m->type,
                             (int)m->in, x, out, m->offset, rb, (int)m->out,
                             n_tokens, x_stride, out_stride);
    if (rc != BATCH_OK) {
        return rc;
    }
    return cudaGetLastError() == cudaSuccess ? BATCH_OK : BATCH_ERR_CUDA;
}

// Block reduction over BATCH_BLOCK_SIZE threads; tmp has one slot per warp.
// Mirrors lcr_block_sum in lcr_kernels.cu (separate translation unit).
__device__ __forceinline__ float batch_block_sum(float v, float *tmp) {
    for (int s = 16; s > 0; s >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, s);
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        tmp[wid] = v;
    }
    __syncthreads();
    float r = threadIdx.x < (BATCH_BLOCK_SIZE / 32) ? tmp[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        for (int s = (BATCH_BLOCK_SIZE / 32) / 2; s > 0; s >>= 1) {
            r += __shfl_down_sync(0xFFFFFFFFu, r, s);
        }
        if (threadIdx.x == 0) {
            tmp[0] = r;
        }
    }
    __syncthreads();
    return tmp[0];
}

// One thread owns one channel across all tokens, so the in-place history
// shift needs no cross-thread synchronization.
__global__ void lcr_batch_conv_silu_kernel(const float *w, float *hist,
                                           const float *qkv, float *out,
                                           int n_tokens, int qkv_stride,
                                           int out_stride, int channels, int k,
                                           int pitch) {
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < channels;
         c += blockDim.x * gridDim.x) {
        for (int t = 0; t < n_tokens; ++t) {
            const float q = qkv[(size_t)t * qkv_stride + c];
            float sum = 0.0f;
            for (int j = 0; j < k - 1; ++j) {
                sum += w[(size_t)c * k + j] * hist[(size_t)j * pitch + c];
            }
            sum += w[(size_t)c * k + (k - 1)] * q;
            out[(size_t)t * out_stride + c] = sum / (1.0f + expf(-sum));
            for (int j = 0; j < k - 2; ++j) {
                hist[(size_t)j * pitch + c] = hist[(size_t)(j + 1) * pitch + c];
            }
            if (k >= 2) {
                hist[(size_t)(k - 2) * pitch + c] = q;
            }
        }
    }
}

int lcr_batch_conv_silu(void *ctx, const float *w, float *hist, const float *qkv,
                        float *out, int n_tokens, int qkv_stride, int out_stride,
                        int channels, int k, int stride) {
    if (!ctx || !w || !qkv || !out || n_tokens <= 0 || channels <= 0 || k < 1 ||
        qkv_stride < channels || out_stride < channels || stride < 0 ||
        (k > 1 && !hist)) {
        return BATCH_ERR_ARG;
    }
    BatchContext *c = (BatchContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return BATCH_ERR_CUDA;
    }
    // Same padded-row convention as lcr_op_conv_silu.
    const int pitch = stride >= channels ? stride : channels;
    const int blocks = (channels + BATCH_BLOCK_SIZE - 1) / BATCH_BLOCK_SIZE;
    lcr_batch_conv_silu_kernel<<<blocks, BATCH_BLOCK_SIZE, 0, c->stream>>>(
        w, hist, qkv, out, n_tokens, qkv_stride, out_stride, channels, k, pitch);
    return cudaGetLastError() == cudaSuccess ? BATCH_OK : BATCH_ERR_CUDA;
}

// One block per value head. Each thread owns whole state rows j, so the
// recurrence needs no cross-thread state synchronization.
__global__ void lcr_batch_gdn_kernel(const float *q, const float *k,
                                     const float *v, const float *gate,
                                     const float *beta, float *state, float *out,
                                     int n_tokens, int q_stride, int kv_stride,
                                     int v_stride, int g_stride, int n_k,
                                     int d) {
    extern __shared__ float smem[];
    const int h = blockIdx.x;
    const int kh = h % n_k;
    const int tid = threadIdx.x;
    float *sk = smem;
    float *sq = smem + d;
    const float inv = rsqrtf((float)d);
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = tid; i < d; i += blockDim.x) {
            sk[i] = k[(size_t)t * kv_stride + (size_t)kh * d + i];
            sq[i] = q[(size_t)t * q_stride + (size_t)kh * d + i];
        }
        if (tid == 0) {
            smem[2 * d] = expf(gate[(size_t)t * g_stride + h]);
            smem[2 * d + 1] = beta[(size_t)t * g_stride + h];
        }
        __syncthreads();
        const float decay = smem[2 * d];
        const float b = smem[2 * d + 1];
        for (int j = tid; j < d; j += blockDim.x) {
            float *row = state + (size_t)h * d * d + (size_t)j * d;
            float kdot = 0.0f;
            for (int i = 0; i < d; ++i) {
                row[i] *= decay;
                kdot += row[i] * sk[i];
            }
            const float delta = (v[(size_t)t * v_stride + (size_t)h * d + j] - kdot) * b;
            float acc = 0.0f;
            for (int i = 0; i < d; ++i) {
                row[i] += delta * sk[i];
                acc += row[i] * sq[i];
            }
            out[(size_t)t * v_stride + (size_t)h * d + j] = acc * inv;
        }
        __syncthreads();
    }
}

int lcr_batch_gdn(void *ctx, const float *q, const float *k, const float *v,
                  const float *gate, const float *beta, float *state, float *out,
                  int n_tokens, int q_stride, int kv_stride, int v_stride,
                  int g_stride, int n_v, int n_k, int d) {
    if (!ctx || !q || !k || !v || !gate || !beta || !state || !out ||
        n_tokens <= 0 || n_v <= 0 || n_k <= 0 || d <= 0 || n_v % n_k != 0 ||
        q_stride < n_k * d || kv_stride < n_k * d || v_stride < n_v * d ||
        g_stride < n_v) {
        return BATCH_ERR_ARG;
    }
    BatchContext *c = (BatchContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return BATCH_ERR_CUDA;
    }
    const size_t smem_bytes = (size_t)(2 * d + 2) * sizeof(float);
    lcr_batch_gdn_kernel<<<n_v, BATCH_BLOCK_SIZE, smem_bytes, c->stream>>>(
        q, k, v, gate, beta, state, out, n_tokens, q_stride, kv_stride,
        v_stride, g_stride, n_k, d);
    return cudaGetLastError() == cudaSuccess ? BATCH_OK : BATCH_ERR_CUDA;
}

// One block per query head. Each token first publishes its k/v into the cache,
// then runs the same online softmax as lcr_attn_kernel over the causal prefix.
__global__ void lcr_batch_attn_kernel(const float *q, const float *k,
                                      const float *v, const float *g,
                                      float *kc, float *vc, float *out,
                                      int n_tokens, int q_stride,
                                      int kv_stride, int out_stride,
                                      int n_heads, int n_kv, int d,
                                      int seq_start, int seq_cap, float scale) {
    __shared__ float tmp[BATCH_BLOCK_SIZE / 32];
    extern __shared__ float acc[];
    const int h = blockIdx.x;
    const int kv = h * n_kv / n_heads;
    const int tid = threadIdx.x;
    const size_t kvbase = (size_t)kv * seq_cap * d;

    for (int t = 0; t < n_tokens; ++t) {
        const int pos = seq_start + t;
        for (int j = tid; j < d; j += BATCH_BLOCK_SIZE) {
            const size_t off = kvbase + (size_t)pos * d + j;
            kc[off] = k[(size_t)t * kv_stride + (size_t)kv * d + j];
            vc[off] = v[(size_t)t * kv_stride + (size_t)kv * d + j];
            acc[j] = 0.0f;
        }
        __syncthreads();

        const float *qt = q + (size_t)t * q_stride + (size_t)h * d;
        float m = -INFINITY;
        float s = 0.0f;
        for (int p = 0; p <= pos; ++p) {
            const float *kt = kc + kvbase + (size_t)p * d;
            const float *vt = vc + kvbase + (size_t)p * d;
            float part = 0.0f;
            for (int j = tid; j < d; j += BATCH_BLOCK_SIZE) {
                part += qt[j] * kt[j];
            }
            const float score = batch_block_sum(part, tmp) * scale;
            const float m_new = fmaxf(m, score);
            const float alpha = expf(m - m_new);
            const float pw = expf(score - m_new);
            for (int j = tid; j < d; j += BATCH_BLOCK_SIZE) {
                acc[j] = acc[j] * alpha + pw * vt[j];
            }
            s = s * alpha + pw;
            m = m_new;
        }
        const float inv = 1.0f / s;
        const float *gt = g + (size_t)t * q_stride + (size_t)h * d;
        float *ot = out + (size_t)t * out_stride + (size_t)h * d;
        for (int j = tid; j < d; j += BATCH_BLOCK_SIZE) {
            const float gate = 1.0f / (1.0f + expf(-gt[j]));
            ot[j] = acc[j] * inv * gate;
        }
    }
}

int lcr_batch_attn(void *ctx, const float *q, const float *k, const float *v,
                   const float *g, float *kc, float *vc, float *out,
                   int n_tokens, int q_stride, int kv_stride, int out_stride,
                   int n_heads, int n_kv, int d, int seq_start, int seq_cap,
                   float scale) {
    if (!ctx || !q || !k || !v || !g || !kc || !vc || !out || n_tokens <= 0 ||
        n_heads <= 0 || n_kv <= 0 || d <= 0 || n_heads % n_kv != 0 ||
        q_stride < n_heads * d || kv_stride < n_kv * d ||
        out_stride < n_heads * d || seq_cap <= 0 || seq_start < 0 ||
        seq_start + n_tokens > seq_cap) {
        return BATCH_ERR_ARG;
    }
    BatchContext *c = (BatchContext *)ctx;
    if (cudaSetDevice(c->device) != cudaSuccess) {
        return BATCH_ERR_CUDA;
    }
    lcr_batch_attn_kernel<<<n_heads, BATCH_BLOCK_SIZE, (size_t)d * sizeof(float),
                            c->stream>>>(
        q, k, v, g, kc, vc, out, n_tokens, q_stride, kv_stride, out_stride,
        n_heads, n_kv, d, seq_start, seq_cap, scale);
    return cudaGetLastError() == cudaSuccess ? BATCH_OK : BATCH_ERR_CUDA;
}
