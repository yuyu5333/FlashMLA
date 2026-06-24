#include <pybind11/pybind11.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    // [Stage-1a] sparse_decode_fwd PYBIND. Bare m.def() relies on
    //   * torch's bundled std::optional<at::Tensor> caster (registered via
    //     torch/csrc/utils/pybind.h, pulled in transitively by ATen headers)
    //   * sparse_attn_decode_interface taking const std::optional<at::Tensor>&
    //     for ALL Tensor optionals (including tile_scheduler_metadata /
    //     num_splits, switched in this commit) so the caster can bind to
    //     a temporary built from Python None.
    // We deliberately do NOT add `#include <pybind11/stl.h>`: stl.h's
    // generic std::optional<T> caster would recursively look up a caster
    // for `at::Tensor`, which is not a pybind11-native type, and dispatch
    // would fail. Torch's specialized caster handles
    // std::optional<at::Tensor> as a single unit and accepts Python None.
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
