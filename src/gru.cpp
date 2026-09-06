// SPDX-License-Identifier: MPL-2.0
#include "gru.h"
#include "gru-tag.h"
#include "ggml-cpu.h"
#include <array>
#include <cmath>
#include <vector>
#ifdef RMVPE_GRU_AVX2
#include <immintrin.h>
#endif
namespace rmvpe {
namespace {
float dot(const float * a, const float * b) {
#ifdef RMVPE_GRU_AVX2
    if (ggml_cpu_has_avx2() && ggml_cpu_has_fma()) {
        __m256 sum = _mm256_setzero_ps();
        for (int k = 0; k < 256; k += 8)
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(a+k), _mm256_loadu_ps(b+k), sum);
        alignas(32) float v[8];
        _mm256_store_ps(v, sum);
        return v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[6]+v[7];
    }
#endif
    float sum = 0;
    for (int k = 0; k < 256; ++k) sum += a[k] * b[k];
    return sum;
}
void compute(ggml_tensor * dst, int ith, int, void * userdata) {
    if (ith) return;
    const bool reverse = userdata != nullptr;
    const auto * x = static_cast<const float *>(dst->src[0]->data);
    const auto * w = static_cast<const float *>(dst->src[1]->data);
    std::vector<float> reordered;
    if (dst->src[1]->ne[0] == 768) {
        reordered.resize(768*256);
        for (int j=0;j<768;++j) for(int k=0;k<256;++k) reordered[j*256+k]=w[k*768+j];
        w = reordered.data();
    }
    const auto * b = static_cast<const float *>(dst->src[2]->data);
    auto * y = static_cast<float *>(dst->data);
    const int frames = static_cast<int>(dst->ne[1]);
    std::array<float, 256> h{}, next{};
    for (int step = 0; step < frames; ++step) {
        int t = reverse ? frames - 1 - step : step;
        const float * p = x + t * 768;
        for (int j = 0; j < 256; ++j) {
            float r = 1.0f / (1.0f + std::exp(-(p[j] + b[j] + dot(w+j*256, h.data()))));
            float z = 1.0f / (1.0f + std::exp(-(p[256+j] + b[256+j] + dot(w+(256+j)*256, h.data()))));
            float n = std::tanh(p[512+j] + r * (b[512+j] + dot(w+(512+j)*256, h.data())));
            next[j] = (1-z)*n + z*h[j];
            y[t*256+j] = next[j];
        }
        h = next;
    }
}
}
ggml_tensor * gru_fused(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, bool reverse) {
    ggml_tensor * args[] = {x, w, b};
    auto * out = ggml_custom_4d(ctx, GGML_TYPE_F32, 256, x->ne[1], 1, 1, args, 3,
                               compute, 1, reverse ? reinterpret_cast<void *>(1) : nullptr);
    out->op_params[RMVPE_GRU_TAG_SLOT] = RMVPE_GRU_TAG;
    out->op_params[RMVPE_GRU_TAG_SLOT+1] = reverse ? 1 : 0;
    return out;
}
}
