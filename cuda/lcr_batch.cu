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

__device__ __forceinline__ float batch_dequant_at(int type, const uint8_t *row, int n) {
    switch (type) {
    case BATCH_F32: {
        float v;
        memcpy(&v, row + (size_t)n * sizeof(float), sizeof(v));
        return v;
    }
    case BATCH_F16: {
        uint16_t h;
        memcpy(&h, row + (size_t)n * sizeof(uint16_t), sizeof(h));
        return batch_half2float(h);
    }
    case BATCH_BF16: {
        uint16_t b;
        memcpy(&b, row + (size_t)n * sizeof(uint16_t), sizeof(b));
        return batch_bf162float(b);
    }
    case BATCH_Q8_0: {
        const BatchBlockQ8_0 *blk =
            (const BatchBlockQ8_0 *)(row + (size_t)(n >> 5) * sizeof(BatchBlockQ8_0));
        return (float)blk->qs[n & 31] * batch_half2float(blk->d);
    }
    case BATCH_Q4_K: {
        const BatchBlockQ4_K *blk =
            (const BatchBlockQ4_K *)(row + (size_t)(n >> 8) * sizeof(BatchBlockQ4_K));
        const int j = n & 255;
        const int il = j >> 6;
        const int is = 2 * il + ((j >> 5) & 1);
        uint8_t sc, m;
        batch_get_scale_min_k4(is, blk->scales, sc, m);
        const float dall = batch_half2float((uint16_t)(blk->dm & 0xFFFF));
        const float dmin = batch_half2float((uint16_t)(blk->dm >> 16));
        const uint8_t q = blk->qs[32 * il + (j & 31)];
        const float d = dall * sc;
        const float dm = dmin * m;
        return (j & 32) ? d * (q >> 4) - dm : d * (q & 0xF) - dm;
    }
    case BATCH_Q5_K: {
        const BatchBlockQ5_K *blk =
            (const BatchBlockQ5_K *)(row + (size_t)(n >> 8) * sizeof(BatchBlockQ5_K));
        const int j = n & 255;
        const int il = j >> 6;
        const int t = j & 63;
        const int upper = t >> 5;
        const int is = 2 * il + upper;
        uint8_t sc, m;
        batch_get_scale_min_k4(is, blk->scales, sc, m);
        const float dall = batch_half2float((uint16_t)(blk->dm & 0xFFFF));
        const float dmin = batch_half2float((uint16_t)(blk->dm >> 16));
        const int qidx = 32 * il + (t & 31);
        const uint8_t qhbit = (uint8_t)(1u << (2 * il + upper));
        const uint8_t qh = blk->qh[t & 31];
        const uint8_t ql = blk->qs[qidx];
        const int q = (upper ? (ql >> 4) : (ql & 0xF)) | ((qh & qhbit) ? 16 : 0);
        return dall * sc * q - dmin * m;
    }
    case BATCH_Q6_K: {
        const BatchBlockQ6_K *blk =
            (const BatchBlockQ6_K *)(row + (size_t)(n >> 8) * sizeof(BatchBlockQ6_K));
        const int j = n & 255;
        const int ip = j >> 7;
        const int t = j & 127;
        const int group = t >> 5;
        const int l = t & 31;
        const int is = 8 * ip + (l >> 4) + 2 * group;
        const uint8_t ql = blk->ql[64 * ip + l + 32 * (group & 1)];
        const uint8_t qh = blk->qh[32 * ip + l] >> (2 * group);
        const uint8_t q4 = (group & 2) ? (ql >> 4) : (ql & 0xF);
        const int q = (int)((int8_t)(q4 | ((qh & 3) << 4)) - 32);
        return batch_half2float(blk->d) * (float)blk->scales[is] * (float)q;
    }
    default:
        return 0.0f;
    }
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

__global__ void lcr_batch_matmul_kernel(const uint8_t *blob, int type, int in,
                                        const float *x, float *out,
                                        int64_t offset, size_t row_bytes,
                                        int x_stride, int out_stride) {
    __shared__ float smem[BATCH_BLOCK_SIZE / 32];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int tok = blockIdx.y;
    const uint8_t *w = blob + offset + (size_t)row * row_bytes;
    const float *xr = x + (size_t)tok * x_stride;

    float sum = 0.0f;
    for (int n = tid; n < in; n += BATCH_BLOCK_SIZE) {
        sum += batch_dequant_at(type, w, n) * xr[n];
    }

    for (int stride = 16; stride > 0; stride >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, stride);
    }
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (lane == 0) {
        smem[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        float v = lane < (BATCH_BLOCK_SIZE / 32) ? smem[lane] : 0.0f;
        for (int stride = 8; stride > 0; stride >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFFu, v, stride);
        }
        if (lane == 0) {
            out[(size_t)tok * out_stride + row] = v;
        }
    }
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

    dim3 grid((unsigned int)m->out, (unsigned int)n_tokens);
    lcr_batch_matmul_kernel<<<grid, BATCH_BLOCK_SIZE, 0, c->stream>>>(
        (const uint8_t *)c->d_blob, (int)m->type, (int)m->in, x, out,
        m->offset, rb, x_stride, out_stride);
    return cudaGetLastError() == cudaSuccess ? BATCH_OK : BATCH_ERR_CUDA;
}
