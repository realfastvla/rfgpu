#ifndef _GRID_H
#define _GRID_H

#include <vector>
#include <map>

#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cusparse.h>

#include "array.h"
#include "timer.h"
#include "device.h"

namespace rfgpu {

    typedef cuComplex cdata; // complex data type
    typedef cufftReal rdata; // real data type

    class Grid : public OnDevice
    {
        public:

            Grid(int nbl, int nchan, int ntime, int upix, int vpix, int device=-1);
            
            ~Grid() {};

            void compute(); // compute/sort gridding matrix
            void conjugate(Array<cdata> &data);
            void downsample(Array<cdata> &data);
            void operate(Array<cdata> &in, Array<cdata> &out, 
                    int itime);
            void operate(cdata *in, cdata *out, int itime);

            void set_cell(float size) { cell = size; }; 
            void set_uv(const std::vector<float> &u, 
                    const std::vector<float> &v);
            void set_freq(const std::vector<float> &freq);
            void set_shift(const std::vector<int> &shift);

            int get_nnz() const { return nnz; }

            IFTIMER( std::map<std::string,Timer *> timers; )

        protected:

            void allocate();  // Alloc memory on GPU

            int ncol() const { return nbl*nchan; }
            int nrow() const { return upix*vpix; }

            std::vector<unsigned> indim() const { 
                std::vector<unsigned> d(3);
                d[0] = nbl;
                d[1] = nchan;
                d[2] = ntime;
                return d;
            }

            std::vector<unsigned> outdim() const { 
                std::vector<unsigned> d(2);
                d[0] = upix;
                d[1] = vpix;
                return d;
            }

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

            // All these point to GPU memory.  Any which should not also be
            // allocated on host should be listed in constructor init list.
            Array<cdata> G_vals;  // Values of gridding matrix G (nnz)
            Array<int> G_rows;    // Row indices, CSR format (nrow+1)
            Array<int> G_cols;    // Column indices, CSR (nnz)
            Array<int> G_chan;    // Channel index of each entry (nnz)
            Array<int> G_cols0;   // Base column indices (nnz)
            Array<int> G_pix;     // pixel index of each entry (nnz)
            Array<int> shift;     // Time shift to apply per channel (nchan)
            Array<int> conj;      // True if baseline needs a conjugate (nbl)
            int maxshift;

            cusparseHandle_t sparse;
            cusparseMatDescr_t descr;

            // Constant values to pass to matrix routines
            float2 h_one;
            float2 h_zero;
    };

}
#endif
