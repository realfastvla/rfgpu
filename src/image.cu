#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <stdexcept>

#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>

#include <cub/cub.cuh>

#include "image.h"

using namespace rfgpu;

Image::Image(int _xpix, int _ypix) {
    plan = 0;
    xpix = _xpix;
    ypix = _ypix;
    setup();
    _stats.resize(2);
}

Image::~Image() {
    if (plan) cufftDestroy(plan);
}

struct fn_square {
    __device__ double operator() (const rdata &x) const { return x*x; }
};

void Image::setup() {
    cufftResult_t rv;
    rv = cufftPlan2d(&plan, xpix, ypix, CUFFT_C2R);
    if (rv != CUFFT_SUCCESS) {
        char msg[1024];
        sprintf(msg, "Image::setup error planning FFT (%d)", rv);
        throw std::runtime_error(msg);
    }

    tmp_bytes_max=0;
    cub::DeviceReduce::Max(NULL, tmp_bytes_max, 
            (rdata *)NULL, _stats.d, imgpix());
    tmp_max.resize(tmp_bytes_max);

    tmp_bytes_sum=0;
    cub::DeviceReduce::Sum(NULL, tmp_bytes_sum,
            cub::TransformInputIterator<double, fn_square, rdata*>(
                (rdata *)NULL, fn_square()),
            _stats.d, imgpix());
    tmp_sum.resize(tmp_bytes_sum);
            
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

void Image::rms(Array<rdata,true> &img) {
    cub::TransformInputIterator<double, fn_square, rdata*> 
        sqr(img.d, fn_square());
    cub::DeviceReduce::Sum(tmp_sum.d, tmp_bytes_sum, sqr, 
            _stats.d + 0, imgpix());
}

void Image::max(Array<rdata,true> &img) {
    cub::DeviceReduce::Max(tmp_max.d, tmp_bytes_max, img.d, 
            _stats.d + 1, imgpix());
}

std::vector<double> Image::stats(Array<rdata,true> &img) {
    rms(img);
    max(img);
    _stats.d2h();
    return std::vector<double>(_stats.h, _stats.h+_stats.len());
}
