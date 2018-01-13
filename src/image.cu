#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>
#include <string>
#include <algorithm>
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
}

Image::~Image() {
    if (plan) cufftDestroy(plan);
    for (unsigned i=0; i<stat_funcs.size(); i++) delete stat_funcs[i];
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

void Image::add_stat(std::string name) {
    if (name=="max") {
        stat_funcs.push_back(new ImageMax(xpix,ypix,name));
    } else if (name=="rms") {
        stat_funcs.push_back(new ImageRMS(xpix,ypix,name));
    } else if (name=="iqr") {
        stat_funcs.push_back(new ImageIQR(xpix,ypix,name));
    } else {
        char msg[1024];
        sprintf(msg, "Image::add_stat unknown stat type '%s'", name.c_str());
        throw std::runtime_error(msg);
    }
    _stats.resize(_stats.len() + 1);
    stat_funcs.back()->setup();
}

std::vector<std::string> Image::stat_names() const {
    std::vector<std::string> names;
    for (unsigned i=0; i<stat_funcs.size(); i++) {
        names.push_back(stat_funcs[i]->name);
    }
    return names;
}

std::vector<double> Image::stats(Array<rdata,true> &img) {
    for (unsigned i=0; i<stat_funcs.size(); i++)
        stat_funcs[i]->operate(img, _stats.d + i);
    _stats.d2h();
    for (unsigned i=0; i<stat_funcs.size(); i++) 
        _stats.h[i] = stat_funcs[i]->finalize(_stats.h[i]);
    return std::vector<double>(_stats.h, _stats.h+_stats.len());
}

ImageStatistic::ImageStatistic(int _xpix, int _ypix, std::string _name) {
    xpix = _xpix;
    ypix = _ypix;
    name = _name;
}

void ImageStatistic::setup() {
    tmp_bytes = 0;
    calc_buffer_size(); // This should fill in value for tmp_bytes
    tmp.resize(tmp_bytes);
}

void ImageMax::calc_buffer_size() {
    cub::DeviceReduce::Max(NULL, tmp_bytes, (rdata *)NULL, (double *)NULL, 
            xpix*ypix);
}

void ImageMax::operate(Array<rdata,true> &img, double *result) {
    cub::DeviceReduce::Max(tmp.d, tmp_bytes, img.d, result, xpix*ypix);
}

struct fn_square {
    __device__ double operator() (const rdata &x) const { return x*x; }
};

void ImageRMS::calc_buffer_size() {
    cub::DeviceReduce::Sum(NULL, tmp_bytes,
            cub::TransformInputIterator<double, fn_square, rdata*>(
                (rdata *)NULL, fn_square()),
            (double *)NULL, xpix*ypix);
}

void ImageRMS::operate(Array<rdata,true> &img, double *result) {
    cub::TransformInputIterator<double, fn_square, rdata*> 
        sqr(img.d, fn_square());
    cub::DeviceReduce::Sum(tmp.d, tmp_bytes, sqr, result, xpix*ypix);
}

void ImageIQR::calc_buffer_size() {
    cub::DeviceRadixSort::SortKeys(NULL, tmp_bytes, 
            (rdata *)NULL, (rdata *)NULL, 
            xpix*ypix);
    tmp_bytes += sizeof(rdata)*xpix*ypix;
}

__global__ void get_iqr(rdata *sorted, double *result, int npix) {
    const int ii = blockDim.x*blockIdx.x + threadIdx.x;
    if (ii==0) *result = sorted[3*npix/4] - sorted[npix/4];
}

void ImageIQR::operate(Array<rdata,true> &img, double *result) {
    size_t offs = sizeof(rdata)*xpix*ypix;
    size_t tmp_bytes_only = tmp_bytes - offs;
    rdata *sorted = (rdata *)(tmp.d + offs);
    cub::DeviceRadixSort::SortKeys(tmp.d, tmp_bytes_only,
            img.d, sorted, xpix*ypix);
    get_iqr<<<1,1>>>(sorted, result, xpix*ypix);
}
