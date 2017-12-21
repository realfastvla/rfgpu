#include <stdio.h>
#include <stdlib.h>

#include <stdexcept>

#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>

#include "image.h"

using namespace rfgpu;

Image::Image() {
    plan = 0;
    xpix = ypix = 0;
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

void Image::operate(cufftComplex *vis, cufftReal *img) {
    cufftResult_t rv;
    rv = cufftExecC2R(plan, vis, img);
    if (rv != CUFFT_SUCCESS) {
        char msg[1024];
        sprintf(msg, "Image::operate error executing FFT (%d)", rv);
        throw std::runtime_error(msg);
    }
}

