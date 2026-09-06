// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "ggml.h"
namespace rmvpe {
ggml_tensor * gru_fused(ggml_context * ctx, ggml_tensor * projected,
                       ggml_tensor * recurrent, ggml_tensor * bias, bool reverse);
}
