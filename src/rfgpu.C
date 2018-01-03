#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <vector>
#include <complex>
#include <cuda.h>

#include "array.h"
#include "grid.h"
#include "image.h"
namespace rf = rfgpu;

typedef rf::Array<rf::cdata,true> GPUArrayComplex;
typedef rf::Array<rf::rdata,true> GPUArrayReal;

PYBIND11_MODULE(rfgpu, m) {

    py::class_<GPUArrayComplex>(m, "GPUArrayComplex")
        .def(py::init())
        .def(py::init<unsigned>())
        .def(py::init<std::vector<unsigned>>())
        .def("resize", &GPUArrayComplex::resize)
        .def("len", &GPUArrayComplex::len)
        .def("h2d", &GPUArrayComplex::h2d)
        .def("d2h", &GPUArrayComplex::d2h)
        .def_property_readonly("data", [](py::object &obj) {
                GPUArrayComplex &a = obj.cast<GPUArrayComplex&>();
                return py::array_t<std::complex<float>>(
                        a.dims(),
                        (std::complex<float> *)a.h,
                        obj
                        );
                });

    py::class_<GPUArrayReal>(m, "GPUArrayReal")
        .def(py::init())
        .def(py::init<unsigned>())
        .def("resize", &GPUArrayReal::resize)
        .def("len", &GPUArrayReal::len)
        .def("h2d", &GPUArrayReal::h2d)
        .def("d2h", &GPUArrayReal::d2h)
        .def_property_readonly("data", [](py::object &obj) {
                GPUArrayReal &a = obj.cast<GPUArrayReal&>();
                return py::array_t<float>(
                        a.dims(),
                        (float *)a.h,
                        obj
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

    py::class_<rf::Image>(m, "Image")
        .def(py::init<int,int>())
        .def("operate",
                (void (rf::Image::*)(GPUArrayComplex&, GPUArrayReal&))
                &rf::Image::operate);
}
