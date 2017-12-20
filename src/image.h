#ifndef _IMAGE_H
#define _IMAGE_H

#include <vector>

#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cusparse.h>

#include "array.h"

namespace rfgpu {

    typedef cuComplex cdata; // complex data type
    typedef cufftReal rdata; // real data type

    class Image
    {
        public:
            Image();
            ~Image();

            void setup();
            void operate(cufftComplex *vis, cufftReal *img);

            int vispix() const { return ((ypix/2)+1) * xpix; }
            int imgpix() const { return xpix*ypix; }

            int xpix; // Number of image pixels
            int ypix; // Numer of image pixels

            cufftHandle plan;
    };

}
#endif
