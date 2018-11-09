#ifndef _ARRAY_H
#define _ARRAY_H

#include <map>
#include <vector>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime_api.h>

// Run any func that returns a cudaError_t, raise exception if needed
#define CUDA_ERROR_CHECK(code) { \
    cudaError_t rv = code; \
    if (rv!=cudaSuccess) { \
        char msg[1024]; \
        sprintf(msg, "Array: '%s' returned '%s'", #code, \
                cudaGetErrorName(rv)); \
        throw std::runtime_error(msg); \
    } \
}

namespace rfgpu {

    template <class T>
    class Array
    {
        public:
            Array(bool _host=true);
            Array(unsigned len, bool _host=true);
            Array(std::vector<unsigned> dims, bool _host=true);
            Array(std::vector<unsigned> dims, std::vector<int> devices, bool _host=true);
            ~Array();
            void set_devices (std::vector<int> devices) { _devices=devices; }
            void resize(unsigned len);
            void resize(std::vector<unsigned> len);
            void free();
            size_t size() const { return sizeof(T)*_len; }
            int len() const { return _len; }
            std::vector<unsigned> dims() const { return _dims; }
            std::vector<int> devices() const { return _devices; }
            bool has_device(int device) const { return dd.find(device)!=dd.end(); }
            T *h; // Pointer to data on host
            T *d; // Pointer to data on first gpu; for backwards compatibility
            std::map<int, T*> dd; // Actual map of device -> pointer
            void h2d(); // Copy data from host to device
            void d2h(int device=-1); // Copy data from device to host
            void init(T val); // Only on host
        protected:
            unsigned _len;
            std::vector<unsigned> _dims;
            std::vector<int> _devices;
            bool host;  // true if array will also be on host
    };

    template <class T>
    Array<T>::Array(bool _host) {
        host = _host;
        _len = 0;
        h = NULL;
        d = NULL;
    }

    template <class T>
    Array<T>::Array(unsigned len, bool _host) { 
        host = _host;
        h = NULL;
        d = NULL;
        resize(len); 
    }

    template <class T>
    Array<T>::Array(std::vector<unsigned> dims, bool _host) { 
        host = _host;
        h = NULL;
        d = NULL;
        resize(dims);
    }

    template <class T>
    Array<T>::Array(std::vector<unsigned> dims, std::vector<int> devices, bool _host) { 
        host = _host;
        h = NULL;
        d = NULL;
        set_devices(devices);
        resize(dims);
    }

    template <class T>
    void Array<T>::resize(unsigned len) {
        std::vector<unsigned> dims;
        dims.push_back(len);
        resize(dims);
    }

    template <class T>
    void Array<T>::resize(std::vector<unsigned> dims) {
        int curdev;
        CUDA_ERROR_CHECK(cudaGetDevice(&curdev));
        if (_devices.empty()) { _devices.push_back(curdev); }
        _len = dims[0];
        for (unsigned i=1; i<dims.size(); i++) { _len *= dims[i]; }
        _dims = dims;
        free();
        if (host) CUDA_ERROR_CHECK(cudaMallocHost((void**)&h, size()));
        for (auto it=_devices.begin(); it!=_devices.end(); ++it) {
            T* dtmp;
            CUDA_ERROR_CHECK(cudaSetDevice(*it));
            CUDA_ERROR_CHECK(cudaMalloc((void**)&dtmp, size()));
            dd[*it] = dtmp;
        }
        d = dd[_devices[0]];
        CUDA_ERROR_CHECK(cudaSetDevice(curdev));
    }

    template <class T>
    void Array<T>::free() {
        if (h) CUDA_ERROR_CHECK(cudaFreeHost(h));
        h = NULL;
        for (auto it=dd.begin(); it!=dd.end(); ++it)
            if (it->second) CUDA_ERROR_CHECK(cudaFree(it->second));
        dd.erase(dd.begin(),dd.end());
    }

    template <class T>
    Array<T>::~Array() {
        free();
    }

    template <class T>
    void Array<T>::h2d() {
        if (!h || !d || dd.empty())  
            throw std::runtime_error("Array::h2d() missing allocation");

        // Save current selected device
        int curdev;
        CUDA_ERROR_CHECK(cudaGetDevice(&curdev));

        // Send data to all devices
        for (auto it=dd.begin(); it!=dd.end(); ++it) {
            CUDA_ERROR_CHECK(cudaSetDevice(it->first));
            CUDA_ERROR_CHECK(cudaMemcpyAsync(it->second, h, size(), 
                        cudaMemcpyHostToDevice));
        }

        // Reset to original device setting
        CUDA_ERROR_CHECK(cudaSetDevice(curdev));
    }

    template <class T>
    void Array<T>::d2h(int device) {
        if (!h || !d || dd.empty())
            throw std::runtime_error("Array::d2h() missing allocation");
        if (device<0) { device = _devices[0]; } 
        if (dd.find(device)==dd.end())
            throw std::runtime_error("Array::d2h() requested device not allocated");

        // Save current selected device
        int curdev;
        CUDA_ERROR_CHECK(cudaGetDevice(&curdev));

        // Do the copy
        CUDA_ERROR_CHECK(cudaSetDevice(device));
        CUDA_ERROR_CHECK(cudaMemcpy(h, dd[device], size(), cudaMemcpyDeviceToHost));

        // Reset to original device setting
        CUDA_ERROR_CHECK(cudaSetDevice(curdev));
    }

    template <class T>
    void Array<T>::init(T val) {
        if (h) { for (int i=0; i<_len; i++) { h[i] = val; } }
    }

}
#endif
