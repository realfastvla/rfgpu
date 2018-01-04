#ifndef _ARRAY_H
#define _ARRAY_H

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

    template <class T, bool host=false>
    class Array
    {
        public:
            Array();
            Array(unsigned len);
            Array(std::vector<unsigned> dims);
            ~Array();
            void resize(unsigned len);
            size_t size() const { return sizeof(T)*_len; }
            int len() const { return _len; }
            std::vector<unsigned> dims() const { return _dims; }
            T *h; // Pointer to data on host
            T *d; // Pointer to data on gpu
            void h2d(); // Copy data from host to device
            void d2h(); // Copy data from device to host
            void init(T val); // Only on host
        protected:
            unsigned _len;
            std::vector<unsigned> _dims;
    };

    template <class T, bool host>
    Array<T,host>::Array() {
        _len = 0;
        h = NULL;
        d = NULL;
    }

    template <class T, bool host>
    Array<T,host>::Array(unsigned len) { 
        h = NULL;
        d = NULL;
        resize(len); 
    }

    template <class T, bool host>
    Array<T,host>::Array(std::vector<unsigned> dims) { 
        h = NULL;
        d = NULL;
        unsigned len = dims[0];
        for (unsigned i=1; i<dims.size(); i++) { len *= dims[i]; }
        resize(len); 
        _dims = dims;
    }

    template <class T, bool host>
    void Array<T,host>::resize(unsigned len) {
        _len = len;
        _dims.resize(1);
        _dims[0] = len;
        if (d) CUDA_ERROR_CHECK(cudaFree(d));
        cudaMalloc((void**)&d, size());
        if (host) {
            if (h) CUDA_ERROR_CHECK(cudaFreeHost(h));
            CUDA_ERROR_CHECK(cudaMallocHost((void**)&h, size()));
        }
    }

    template <class T, bool host>
    Array<T,host>::~Array() {
        if (d) CUDA_ERROR_CHECK(cudaFree(d)); 
        if (h) CUDA_ERROR_CHECK(cudaFreeHost(h));
    }

    template <class T, bool host>
    void Array<T,host>::h2d() {
        if (!h || !d)  
            throw std::runtime_error("Array::h2d() missing allocation");
        CUDA_ERROR_CHECK(cudaMemcpy(d, h, size(), cudaMemcpyHostToDevice));
    }

    template <class T, bool host>
    void Array<T,host>::d2h() {
        if (!h || !d)
            throw std::runtime_error("Array::d2h() missing allocation");
        CUDA_ERROR_CHECK(cudaMemcpy(h, d, size(), cudaMemcpyDeviceToHost));
    }

    template <class T, bool host>
    void Array<T,host>::init(T val) {
        if (h) { for (int i=0; i<_len; i++) { h[i] = val; } }
    }

}
#endif
