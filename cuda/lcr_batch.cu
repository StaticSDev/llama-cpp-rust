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
