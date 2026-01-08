#include "kernel_nanobind_helpers.h"
#include "solver_kernels.h"

namespace eigh {
namespace cuda {

namespace nb = nanobind;

nb::dict Registrations() {
    nb::dict dict;

    dict["eigh_cusolver_sygvd_ffi"] = EncapsulateFfiHandler(SygvdFfi);
    return dict;
}

NB_MODULE(eigh_cuda, m) {
    m.def("registrations", &Registrations);
}

} // namespace cuda
} // namespace eigh
