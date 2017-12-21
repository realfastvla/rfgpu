#include <stdio.h>
#include <stdlib.h>

#include <stdexcept>

#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>

#include "image.h"

using namespace rfgpu;

Image::Image(int _xpix, int _ypix) {
    plan = 0;
    xpix = _xpix;
    ypix = _ypix;
    setup();
}

Image::~Image() {
    if (plan) cufftDestroy(plan);
}

void Image::setup() {
    cufftResult_t rv;
    rv = cufftPlan2d(&plan, xpix, ypix, CUFFT_C2R);
    if (rv != CUFFT_SUCCESS) {
        char msg[1024];
        sprintf(msg, "Image::setup error planning FFT (%d)", rv);
        throw std::runtime_error(msg);
    }
}

void Image::operate(Array<cdata,true> &vis, Array<rdata,true> &img) {
    if (vis.len() != vispix()) {
        char msg[1024];
        sprintf(msg, "Image::operate vis array size (%d) != expected (%d)",
                vis.len(), vispix());
        throw std::invalid_argument(msg);
    }
    if (img.len() != imgpix()) {
        char msg[1024];
        sprintf(msg, "Image::operate img array size (%d) != expected (%d)",
                img.len(), imgpix());
        throw std::invalid_argument(msg);
    }
    operate(vis.d, img.d);
}

void Image::operate(cdata *vis, rdata *img) {
    cufftResult_t rv;
    rv = cufftExecC2R(plan, vis, img);
    if (rv != CUFFT_SUCCESS) {
        char msg[1024];
        sprintf(msg, "Image::operate error executing FFT (%d)", rv);
        throw std::runtime_error(msg);
    }
}

