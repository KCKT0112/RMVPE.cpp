// SPDX-License-Identifier: MPL-2.0
// Private ggml v0.19.0 custom-op contract; does not change ggml's public ABI.
#pragma once
#include "ggml.h"
constexpr int RMVPE_GRU_TAG_SLOT = 12; // byte 48, beyond the CPU callback payload
constexpr int RMVPE_GRU_TAG = 0x52564731;
inline bool rmvpe_gru_valid(const ggml_tensor * t) {
    if (!t || t->op != GGML_OP_CUSTOM || t->op_params[RMVPE_GRU_TAG_SLOT] != RMVPE_GRU_TAG ||
        t->op_params[RMVPE_GRU_TAG_SLOT+1] < 0 || t->op_params[RMVPE_GRU_TAG_SLOT+1] > 1 ||
        t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != 256 || t->ne[1] < 1 || t->ne[2] != 1 || t->ne[3] != 1 || t->src[3]) return false;
    for (int i=0;i<3;++i) if (!t->src[i] || t->src[i]->type != GGML_TYPE_F32 || !ggml_is_contiguous(t->src[i]) || t->src[i]->ne[2] != 1 || t->src[i]->ne[3] != 1) return false;
    return t->src[0]->ne[0] == 768 && t->src[0]->ne[1] == t->ne[1] &&
        ((t->src[1]->ne[0] == 256 && t->src[1]->ne[1] == 768) || (t->src[1]->ne[0] == 768 && t->src[1]->ne[1] == 256)) &&
        t->src[2]->ne[0] == 768 && t->src[2]->ne[1] == 1;
}
