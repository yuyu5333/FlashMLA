#include <pybind11/pybind11.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    // [Stage-1a fix] Explicit py::arg + py::none() defaults teach pybind11's
    // std::optional caster to accept Python None for the 13 optional kwargs
    // (5 original + 2 sched-meta + 6 packed-FP8). Belt-and-braces with the
    // const-ref change in sparse_decode.h sparse_attn_decode_interface().
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface,
          py::arg("q"),
          py::arg("kv"),
          py::arg("indices"),
          py::arg("topk_length")             = py::none(),
          py::arg("attn_sink")               = py::none(),
          py::arg("tile_scheduler_metadata") = py::none(),
          py::arg("num_splits")              = py::none(),
          py::arg("extra_kv")                = py::none(),
          py::arg("extra_indices")           = py::none(),
          py::arg("extra_topk_length")       = py::none(),
          py::arg("d_v"),
          py::arg("sm_scale"),
          py::arg("packed_kcache")           = py::none(),
          py::arg("scale_kcache")            = py::none(),
          py::arg("R_matrix")                = py::none(),
          py::arg("zero_point")              = py::none(),
          py::arg("dim_of_bit")              = py::none(),
          py::arg("bitpos_in_dim")           = py::none());
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
