#ifndef _GRID_H
#define _GRID_H

#include <vector>

#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cusparse.h>

#include "array.h"

namespace rfgpu {

    typedef cuComplex cdata; // complex data type
    typedef cufftReal rdata; // real data type

    class Grid
    {
        public:

            Grid(int nbl, int nchan, int ntime, int upix, int vpix);
            
            ~Grid() {};

            void compute(); // compute/sort gridding matrix
            void operate(Array<cdata,true> &in, Array<cdata,true> &out, 
                    int itime) { operate(in.d, out.d, itime); };
            void operate(cdata *in, cdata *out, int itime);

            void set_cell(float size) { cell = size; }; 
            void set_uv(const std::vector<float> &u, 
                    const std::vector<float> &v);
            void set_freq(const std::vector<float> &freq);
            void set_shift(const std::vector<int> &shift);

        protected:

            void allocate();  // Alloc memory on GPU

            int ncol() const { return nbl*nchan; }
            int nrow() const { return upix*vpix; }

            int nbl;
            int nchan;
            int ntime;
            int upix;
            int vpix;
            int nnz; // at most equal to ncol
            float cell;

            std::vector<float> u; // us
            std::vector<float> v; // us
            std::vector<float> freq; // MHz

            // All these point to GPU memory:
            Array<cdata> G_vals;  // Values of gridding matrix G (nnz)
            Array<int> G_rows;    // Row indices, CSR format (nrow+1)
            Array<int> G_cols;    // Column indices, CSR (nnz)
            Array<int> G_chan;    // Channel index of each entry (nnz)
            Array<int,true> G_cols0;   // Base column indices (nnz)
            Array<int,true> G_pix;     // pixel index of each entry (nnz)
            Array<int> shift;     // Time shift to apply per channel (nchan)
            int maxshift;

            cusparseHandle_t sparse;
            cusparseMatDescr_t descr;

            // Constant values to pass to matrix routines
            float2 h_one;
            float2 h_zero;
    };

}
#endif
