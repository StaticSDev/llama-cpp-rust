#ifndef LCR_BATCH_CUH
#define LCR_BATCH_CUH

#ifdef __cplusplus
extern "C" {
#endif

/* Batched weight matmul on device pointers.
 *
 *   out[t * out_stride + r] = sum_i W[index][r][i] * x[t * x_stride + i]
 *
 * Launched on the context stream; returns without synchronizing. The caller
 * must issue one lcr_gpu_sync before reading results on the host.
 */
int lcr_batch_matmul(void *ctx, int index, const float *x, float *out,
                     int n_tokens, int x_stride, int out_stride);

/* Sequential-state ops for prompt prefill. Each launches on the context stream
 * and returns without synchronizing. The caller issues one lcr_gpu_sync before
 * reading results on the host.
 *
 * lcr_batch_conv_silu applies lcr_op_conv_silu to qkv[t * qkv_stride], writes
 * out[t * out_stride] and updates hist in place after every token. The w layout
 * is [channels][k]; hist is [k-1][channels] with row pitch `stride` (a stride
 * smaller than channels means contiguous).
 */
int lcr_batch_conv_silu(void *ctx, const float *w, float *hist, const float *qkv,
                        float *out, int n_tokens, int qkv_stride, int out_stride,
                        int channels, int k, int stride);

/* One block per value head, sequential over tokens; state is [n_v][d][d].
 * q/k/v/out are [t * stride + h * d + j]; gate/beta are [t * g_stride + h].
 * The output uses v_stride as its token stride.
 */
int lcr_batch_gdn(void *ctx, const float *q, const float *k, const float *v,
                  const float *gate, const float *beta, float *state, float *out,
                  int n_tokens, int q_stride, int kv_stride, int v_stride,
                  int g_stride, int n_v, int n_k, int d);

/* One block per query head, sequential over tokens. Each token stores its
 * k/v at [n_kv][seq_cap][d] slot seq_start+t, then runs the causal online
 * softmax of lcr_op_attn over slots 0..seq_start+t. q/g are
 * [t * q_stride + h * d + j], out is [t * out_stride + h * d + j].
 */
int lcr_batch_attn(void *ctx, const float *q, const float *k, const float *v,
                   const float *g, float *kc, float *vc, float *out,
                   int n_tokens, int q_stride, int kv_stride, int out_stride,
                   int n_heads, int n_kv, int d, int seq_start, int seq_cap,
                   float scale);

#ifdef __cplusplus
}
#endif

#endif /* LCR_BATCH_CUH */
