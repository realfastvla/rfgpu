#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <vector>
#include <map>
#include <string>
#include <stdexcept>

#include <complex>
#include <cuda.h>

#include "array.h"
#include "grid.h"
#include "image.h"
#include "timer.h"
namespace rf = rfgpu;

typedef rf::Array<rf::cdata> GPUArrayComplex;
typedef rf::Array<rf::rdata> GPUArrayReal;

namespace rfgpu {
    void cudaSetDevice(int device) {
        cudaError_t rv;
        rv = ::cudaSetDevice(device);
        if (rv!=cudaSuccess) {
            char msg[1024];
            sprintf(msg, "cudaSetDevice returned %d", rv);
            throw std::runtime_error(msg);
        }
    }
}

PYBIND11_MODULE(rfgpu, m) {

    m.def("cudaSetDevice", &rf::cudaSetDevice);

    py::class_<GPUArrayComplex>(m, "GPUArrayComplex",
            "Array of complex64 that exists in both host and device memory")
        .def(py::init<unsigned>(), py::arg("length"))
        .def(py::init<std::vector<unsigned>>(), py::arg("dims"))
        .def(py::init<std::vector<unsigned>,std::vector<int>>(),
                py::arg("dims"), py::arg("devices"))
        .def("len", &GPUArrayComplex::len, "Total length of array")
        .def("h2d", &GPUArrayComplex::h2d, "Transfer data from host to device(s)")
        .def("d2h", &GPUArrayComplex::d2h, "Transfer data from a device to host",
                py::arg("device")=-1)
        .def("devices", &GPUArrayComplex::devices, "List of associated devices")
        .def_property_readonly("data", [](py::object &obj) {
                GPUArrayComplex &a = obj.cast<GPUArrayComplex&>();
                return py::array_t<std::complex<float>>(
                        a.dims(),
                        (std::complex<float> *)a.h,
                        obj
                        );
                }, "Numpy view of the array in host memory");

    py::class_<GPUArrayReal>(m, "GPUArrayReal",
            "Array of float32 that exists in both host and device memory")
        .def(py::init<unsigned>(), py::arg("length"))
        .def(py::init<std::vector<unsigned>>(), py::arg("dims"))
        .def(py::init<std::vector<unsigned>,std::vector<int>>(),
                py::arg("dims"), py::arg("devices"))
        .def("len", &GPUArrayReal::len, "Total length of array")
        .def("h2d", &GPUArrayReal::h2d, "Transfer data from host to device(s)")
        .def("d2h", &GPUArrayReal::d2h, "Transfer data from a device to host",
                py::arg("device")=-1)
        .def("devices", &GPUArrayReal::devices, "List of associated devices")
        .def_property_readonly("data", [](py::object &obj) {
                GPUArrayReal &a = obj.cast<GPUArrayReal&>();
                return py::array_t<float>(
                        a.dims(),
                        (float *)a.h,
                        obj
                        );
                }, "Numpy view of the array in host memory");

    py::class_<rf::Grid>(m, "Grid")
        .def(py::init<int,int,int,int,int>())
        .def(py::init<int,int,int,int,int,int>())
        .def("set_uv", &rf::Grid::set_uv)
        .def("set_freq", &rf::Grid::set_freq)
        .def("set_shift", &rf::Grid::set_shift)
        .def("set_cell", &rf::Grid::set_cell)
        .def("compute", &rf::Grid::compute)
        .def("get_nnz", &rf::Grid::get_nnz)
        .def("conjugate", 
                (void (rf::Grid::*)(GPUArrayComplex&))
                &rf::Grid::conjugate)
        .def("downsample", 
                (void (rf::Grid::*)(GPUArrayComplex&))
                &rf::Grid::downsample)
        .def("operate", 
                (void (rf::Grid::*)(GPUArrayComplex&, GPUArrayComplex&, int)) 
                &rf::Grid::operate,
                py::call_guard<py::gil_scoped_release>())
#ifdef USETIMER
        .def("timers", [](py::object &o, bool total) {
                rf::Grid &g = o.cast<rf::Grid&>();
                std::map<std::string,double> result;
                for (std::map<std::string,rf::Timer*>::iterator it=g.timers.begin();
                        it!=g.timers.end(); ++it) 
                    if (total)
                        result[it->first] = it->second->get_time_total();
                    else
                        result[it->first] = it->second->get_time_percall();
                return result;
                }, py::arg("total")=false)
#endif
    ;

    py::class_<rf::Image>(m, "Image")
        .def(py::init<int,int>())
        .def(py::init<int,int,int>())
        .def("operate",
                (void (rf::Image::*)(GPUArrayComplex&, GPUArrayReal&))
                &rf::Image::operate,
                py::call_guard<py::gil_scoped_release>())
        .def("add_stat", &rf::Image::add_stat)
        .def("stats", [](py::object &o, GPUArrayReal &img) {
                rf::Image &i = o.cast<rf::Image&>();
                std::vector<std::string> keys = i.stat_names();
                std::vector<double> vals = i.stats(img);
                std::map<std::string,double> result;
                for (unsigned ii=0; ii<vals.size(); ii++)
                    result[keys[ii]] = vals[ii];
                return result;
                })
#ifdef USETIMER
        .def("timers", [](py::object &o, bool total) {
                rf::Image &i = o.cast<rf::Image&>();
                std::map<std::string,double> result;
                for (std::map<std::string,rf::Timer*>::iterator it=i.timers.begin();
                        it!=i.timers.end(); ++it) 
                    if (total)
                        result[it->first] = it->second->get_time_total();
                    else
                        result[it->first] = it->second->get_time_percall();
                return result;
                }, py::arg("total")=false)
#endif
        ;

}
