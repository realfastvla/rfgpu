#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <complex>
#include <cuda.h>

#include "array.h"
#include "rfgpu.h"
namespace rf = rfgpu;

typedef rf::Array<cuComplex,true> GPUArrayComplex;

PYBIND11_MODULE(rfgpu_py, m) {
    py::class_<GPUArrayComplex>(m, "GPUArrayComplex", py::buffer_protocol())
        .def(py::init())
        .def("resize", &GPUArrayComplex::resize)
        .def("len", &GPUArrayComplex::len)
        .def("h2d", &GPUArrayComplex::h2d)
        .def("d2h", &GPUArrayComplex::d2h)
        .def_buffer([](GPUArrayComplex &m) -> py::buffer_info {
                return py::buffer_info(
                        m.h,
                        sizeof(cuComplex),
                        py::format_descriptor<std::complex<float>>::format(),
                        1,
                        { m.len(), },
                        { sizeof(cuComplex), }
                );
            });


}
