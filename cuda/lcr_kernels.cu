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

// One block per output row, 256 threads. Block reductions in shared memory.
__global__ void lcr_matvec_kernel(const uint8_t *blob, int type, int in,
                                  const float *x, float *out,
                                  int64_t offset, size_t row_bytes) {
    __shared__ float smem[LCR_BLOCK_SIZE / 32];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const uint8_t *w = blob + offset + (size_t)row * row_bytes;

    float sum = 0.0f;
    for (int n = tid; n < in; n += LCR_BLOCK_SIZE) {
        sum += lcr_dequant_at(type, w, n) * x[n];
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
        float v = lane < (LCR_BLOCK_SIZE / 32) ? smem[lane] : 0.0f;
        for (int stride = 8; stride > 0; stride >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFFu, v, stride);
        }
        if (lane == 0) {
            out[row] = v;
        }
    }
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
    *ctx_out = c;
    return LCR_OK;
}

void lcr_gpu_free(void *ctx) {
    if (!ctx) {
        return;
    }
    LcrContext *c = (LcrContext *)ctx;
    cudaSetDevice(c->device);
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
    if (cudaMemcpy(d_blob, blob, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
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

    if (cudaMemcpy(c->d_x, x, nf * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    lcr_matvec_kernel<<<(unsigned int)no, LCR_BLOCK_SIZE>>>(
        (const uint8_t *)c->d_blob, m->type, (int)m->in, c->d_x, c->d_y,
        m->offset, rb);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    if (cudaMemcpy(out, c->d_y, no * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
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

    lcr_row_kernel<<<1, LCR_BLOCK_SIZE>>>(
        (const uint8_t *)c->d_blob, m->type, (int)m->in, c->d_y,
        m->offset + (int64_t)row * (int64_t)rb, rb);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        return LCR_ERR_CUDA;
    }
    if (cudaMemcpy(out, c->d_y, nf * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
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
