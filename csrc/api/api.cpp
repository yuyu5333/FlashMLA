#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // [Stage-1a] required for std::optional<at::Tensor> caster

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    // [Stage-1a] sparse_decode_fwd: bare m.def() + #include <pybind11/stl.h>
    // pybind11's stl.h registers the std::optional<T> caster. Combined with
    // sparse_attn_decode_interface taking const std::optional<at::Tensor>&
    // (changed from non-const ref earlier in this branch), Python None
    // correctly binds via the caster's empty-optional path.
    // No py::arg defaults: pybind11 forbids mandatory args after default-
    // valued args, and the Python wrapper always passes all 18 args
    // positionally, so defaults are unnecessary.
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
