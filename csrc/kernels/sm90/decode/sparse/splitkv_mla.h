#pragma once

#include "kernels/params.h"

namespace sm90::decode::sparse {

template<ModelType MODEL_TYPE, int NUM_HEADS, ModelType EXTRA_MODEL_TYPE = MODEL_TYPE>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);

}
