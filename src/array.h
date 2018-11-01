#ifndef _ARRAY_H
#define _ARRAY_H

#include <set>
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
            ~Array();
            void resize(unsigned len);
            size_t size() const { return sizeof(T)*_len; }
            int len() const { return _len; }
            std::vector<unsigned> dims() const { return _dims; }
            T *h; // Pointer to data on host
            T *d; // Pointer to data on gpu
            // TODO use std::map to have a set of device pointers
            // on specific devices.
            //std::map<int, T*> d;
            void h2d(); // Copy data from host to device
            void d2h(); // Copy data from device to host
            void init(T val); // Only on host
        protected:
            unsigned _len;
            std::vector<unsigned> _dims;
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
        unsigned len = dims[0];
        for (unsigned i=1; i<dims.size(); i++) { len *= dims[i]; }
        resize(len); 
        _dims = dims;
    }

    template <class T>
    void Array<T>::resize(unsigned len) {
        _len = len;
        _dims.resize(1);
        _dims[0] = len;
        if (d) CUDA_ERROR_CHECK(cudaFree(d));
        CUDA_ERROR_CHECK(cudaMalloc((void**)&d, size()));
        if (host) {
            if (h) CUDA_ERROR_CHECK(cudaFreeHost(h));
            CUDA_ERROR_CHECK(cudaMallocHost((void**)&h, size()));
        }
    }

    template <class T>
    Array<T>::~Array() {
        if (d) CUDA_ERROR_CHECK(cudaFree(d)); 
        if (h) CUDA_ERROR_CHECK(cudaFreeHost(h));
    }

    template <class T>
    void Array<T>::h2d() {
        if (!h || !d)  
            throw std::runtime_error("Array::h2d() missing allocation");
        CUDA_ERROR_CHECK(cudaMemcpy(d, h, size(), cudaMemcpyHostToDevice));
    }

    template <class T>
    void Array<T>::d2h() {
        if (!h || !d)
            throw std::runtime_error("Array::d2h() missing allocation");
        CUDA_ERROR_CHECK(cudaMemcpy(h, d, size(), cudaMemcpyDeviceToHost));
    }

    template <class T>
    void Array<T>::init(T val) {
        if (h) { for (int i=0; i<_len; i++) { h[i] = val; } }
    }

}
#endif
