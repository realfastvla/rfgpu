#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <complex>
#include <cuda.h>

#include "array.h"
#include "grid.h"
namespace rf = rfgpu;

typedef rf::Array<rf::cdata,true> GPUArrayComplex;

PYBIND11_MODULE(rfgpu, m) {
    // Note, get a numpy array view into the databuf like:
    // a = rfgpu_py.GPUArrayComplex()
    // aa = numpy.array(a,copy=False)
    py::class_<GPUArrayComplex>(m, "GPUArrayComplex", py::buffer_protocol())
        .def(py::init())
        .def(py::init<unsigned>())
        .def("resize", &GPUArrayComplex::resize)
        .def("len", &GPUArrayComplex::len)
        .def("h2d", &GPUArrayComplex::h2d)
        .def("d2h", &GPUArrayComplex::d2h)
        .def_buffer([](GPUArrayComplex &m) -> py::buffer_info {
                return py::buffer_info(
                        m.h,
                        sizeof(rf::cdata),
                        py::format_descriptor<std::complex<float>>::format(),
                        1,
                        { m.len(), },
                        { sizeof(rf::cdata), }
                );
            });

    py::class_<rf::Grid>(m, "Grid")
        .def(py::init<int,int,int,int,int>())
        .def("set_uv", &rf::Grid::set_uv)
        .def("set_freq", &rf::Grid::set_freq)
        .def("set_shift", &rf::Grid::set_shift)
        .def("set_cell", &rf::Grid::set_cell)
        .def("compute", &rf::Grid::compute)
        .def("operate", 
                (void (rf::Grid::*)(GPUArrayComplex&, GPUArrayComplex&, int)) 
                &rf::Grid::operate);
}
