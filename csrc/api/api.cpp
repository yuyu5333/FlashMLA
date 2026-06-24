#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // [Stage-1a] required for std::optional<at::Tensor> caster

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    namespace py = pybind11;
    // [Stage-1a] sparse_decode_fwd:
    //   * #include <pybind11/stl.h> registers std::optional<T> caster
    //   * sparse_attn_decode_interface uses const std::optional<at::Tensor>&
    //   * Trailing 6 packed-FP8 kwargs get py::arg(name)=py::none() so
    //     pre-Stage-1a callers can keep using 12-arg form.
    //   * pybind11 forbids mandatory args AFTER default-valued args, so
    //     we only attach defaults to args 13-18 (after d_v/sm_scale).
    //     Args 4-10 (the original optional Tensors) stay mandatory; the
    //     Python wrapper always passes None for them positionally, so
    //     no default is needed there.
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
          py::arg("bitpos_in_dim")  = py::none());
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
