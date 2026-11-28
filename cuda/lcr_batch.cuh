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

#ifdef __cplusplus
}
#endif

#endif /* LCR_BATCH_CUH */
