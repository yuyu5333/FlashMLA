// [Stage-1a] Order matters:
//   1. <torch/extension.h> first registers torch's specialized
//      std::optional<at::Tensor> type_caster (and at::Tensor caster).
//   2. Bare pybind11.h is enough afterwards; do NOT include
//      <pybind11/stl.h> — its generic std::optional<T> caster would
//      shadow torch's specialized one and break Tensor binding.
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    namespace py = pybind11;
    // [Stage-1a] sparse_decode_fwd:
    //   * <torch/extension.h> above registers torch's specialized
    //     std::optional<at::Tensor> caster (accepts Python None).
    //   * sparse_attn_decode_interface uses const std::optional<at::Tensor>&
    //   * Trailing 6 packed-FP8 kwargs get py::none() defaults so
    //     pre-Stage-1a 12-arg callers still work.
    //   * pybind11 forbids mandatory args after default-valued args, so
    //     defaults are only attached to args 13-18 (after d_v/sm_scale).
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface,
          py::arg("q"), py::arg("kv"), py::arg("indices"),
          py::arg("topk_length"),
          py::arg("attn_sink"),
          py::arg("tile_scheduler_metadata"),
          py::arg("num_splits"),
          py::arg("extra_kv"),
          py::arg("extra_indices"),
          py::arg("extra_topk_length"),
          py::arg("d_v"), py::arg("sm_scale"),
          py::arg("packed_kcache")  = py::none(),
          py::arg("scale_kcache")   = py::none(),
          py::arg("R_matrix")       = py::none(),
          py::arg("zero_point")     = py::none(),
          py::arg("dim_of_bit")     = py::none(),
          py::arg("bitpos_in_dim")  = py::none(),
          py::arg("bit_uniform")    = (int64_t)0,
          py::arg("q_for_extra")    = py::none(),
          py::arg("q_nope_is_folded") = false);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
