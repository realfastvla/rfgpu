#ifndef _IMAGE_H
#define _IMAGE_H

#include <vector>

#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>

#include "array.h"

namespace rfgpu {

    typedef cuComplex cdata; // complex data type
    typedef cufftReal rdata; // real data type

    class Image
    {
        public:
            Image(int xpix, int ypix);
            ~Image();

            void operate(Array<cdata,true> &vis, Array<rdata,true> &img);

            void operate(cdata *vis, rdata *img);

        protected:
            void setup();

            int vispix() const { return ((ypix/2)+1) * xpix; }
            int imgpix() const { return xpix*ypix; }

            int xpix; // Number of image pixels
            int ypix; // Numer of image pixels

            cufftHandle plan;
    };

}
#endif
