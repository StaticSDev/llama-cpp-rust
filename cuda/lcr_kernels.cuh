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

#ifdef __cplusplus
}
#endif

#endif /* LCR_KERNELS_CUH */
