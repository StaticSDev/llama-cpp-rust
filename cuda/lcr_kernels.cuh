#ifndef LCR_KERNELS_CUH
#define LCR_KERNELS_CUH

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LcrWeightMeta {
    int32_t type;   /* GGML type id */
    int64_t in;     /* in features  (ne[0]) */
    int64_t out;    /* out features (ne[1]) */
    int64_t offset; /* byte offset into blob */
} LcrWeightMeta;

int  lcr_gpu_init(void **ctx_out, int device);
void lcr_gpu_free(void *ctx);
int  lcr_gpu_upload(void *ctx, const void *blob, size_t bytes,
                    const LcrWeightMeta *metas, int count);
int  lcr_gpu_matvec(void *ctx, int index, const float *x, float *out);
int  lcr_gpu_row(void *ctx, int index, int row, float *out);
int  lcr_gpu_info(void *ctx, char *buf, int len);
/* Wait for all work queued on the context stream (call once per token). */
int  lcr_gpu_sync(void *ctx);

/* Device-resident helpers. lcr_gpu_matvec/lcr_gpu_row above keep their host
 * pointer arguments; the variants below never copy to or from the host. */
int  lcr_gpu_zero(void *ctx, void *ptr, size_t bytes);
int  lcr_gpu_row_device(void *ctx, int index, int row, float *dst);
int  lcr_gpu_matvec_device(void *ctx, int index, const float *x, float *out);
/* Grouped matvecs: N weights share one device activation. All arguments are
 * device pointers; each call launches on the context stream and returns
 * without synchronizing. */
int  lcr_gpu_matvec2(void *ctx, int i0, int i1, const float *x,
                     float *o0, float *o1);
int  lcr_gpu_matvec3(void *ctx, int i0, int i1, int i2, const float *x,
                     float *o0, float *o1, float *o2);
int  lcr_gpu_matvec4(void *ctx, int i0, int i1, int i2, int i3, const float *x,
                     float *o0, float *o1, float *o2, float *o3);
int  lcr_gpu_read_i32(void *ctx, const void *src, int *dst);

/* Device scratch allocation and direct access to the uploaded blob. */
int  lcr_gpu_buf(void *ctx, size_t bytes, void **out);
void lcr_gpu_buf_free(void *ctx, void *ptr);
void *lcr_gpu_weight_ptr(void *ctx, int index);

/* Every pointer argument of the lcr_op_* functions below is a DEVICE pointer.
 * Each call launches on the context stream and returns without synchronizing;
 * call lcr_gpu_sync once to wait for the whole token. */
int lcr_op_rmsnorm(void *ctx, const float *x, const float *w, float *y, int n, float eps);
int lcr_op_l2norm(void *ctx, float *x, int n, float eps);
int lcr_op_rope(void *ctx, float *x, int n, int n_rot, float base, int pos);
int lcr_op_add(void *ctx, float *dst, const float *src, int n);
int lcr_op_mul(void *ctx, float *dst, const float *src, int n);
int lcr_op_scale(void *ctx, float *x, int n, float scale);
int lcr_op_sigmoid(void *ctx, const float *x, float *y, int n);
int lcr_op_gate(void *ctx, const float *alpha, const float *bias, const float *a,
                float *gate, int n);
int lcr_op_silu_mul(void *ctx, const float *a, const float *b, float *y, int n);
int lcr_op_conv_silu(void *ctx, const float *w, const float *hist, const float *qkv,
                     float *out, int channels, int k, int stride);
int lcr_op_gdn(void *ctx, const float *q, const float *k, const float *v,
               const float *gate, const float *beta, float *state, float *out,
               int n_v, int n_k, int d);
int lcr_op_attn(void *ctx, const float *q, const float *k, const float *v,
                const float *g, float *kc, float *vc, float *out,
                int n_heads, int n_kv, int d, int seq_pos, int seq_cap, float scale);
int lcr_op_argmax(void *ctx, const float *x, int n, int *out);

/* Fused per-head kernels. One launch covers all heads, no sync on return. */
int lcr_op_split_qg(void *ctx, const float *qg, float *q, float *g,
                    int n_heads, int head_dim);
int lcr_op_qk_norm_rope(void *ctx, float *x, const float *w,
                        int n_heads, int head_dim, int n_rot, float base, int pos);
int lcr_op_l2norm_heads(void *ctx, float *x, int n_heads, int head_dim, float eps);
int lcr_op_gdn_post(void *ctx, const float *gdn, const float *z, const float *norm_w,
                    float *y, int n_heads, int head_dim, float eps);

#ifdef __cplusplus
}
#endif

#endif /* LCR_KERNELS_CUH */
