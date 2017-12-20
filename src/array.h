#ifndef _ARRAY_H
#define _ARRAY_H

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace rfgpu {

    template <class T, bool host=false>
    class Array
    {
        public:
            Array();
            Array(unsigned len);
            ~Array();
            void resize(unsigned len);
            size_t size() const { return sizeof(T)*_len; }
            int len() const { return _len; }
            T *h; // Pointer to data on host
            T *d; // Pointer to data on gpu
            void h2d(); // Copy data from host to device
            void d2h(); // Copy data from device to host
            void init(T val); // Only on host
        protected:
            unsigned _len;
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
    void Array<T,host>::resize(unsigned len) {
        _len = len;
        if (d) cudaFree(d);
        cudaMalloc((void**)&d, size());
        if (host) {
            if (h) cudaFreeHost(h);
            cudaMallocHost((void**)&h, size());
        }
    }

    template <class T, bool host>
    Array<T,host>::~Array() {
        if (h) cudaFree(h); 
        if (d) cudaFreeHost(d);
    }

    template <class T, bool host>
    void Array<T,host>::h2d() {
        if (!h || !d) { } // TODO
        cudaMemcpy(d, h, size(), cudaMemcpyHostToDevice);
    }

    template <class T, bool host>
    void Array<T,host>::d2h() {
        if (!h || !d) { } // TODO
        cudaMemcpy(h, d, size(), cudaMemcpyDeviceToHost);
    }

    template <class T, bool host>
    void Array<T,host>::init(T val) {
        if (h) { for (int i=0; i<_len; i++) { h[i] = val; } }
    }

}
#endif
