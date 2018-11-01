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
#include "timer.h"

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
#ifdef USETIMER
    for (std::map<std::string,Timer*>::iterator i=timers.begin(); 
            i!=timers.end(); i++) delete i->second;
#endif
}

void Image::setup() {
    cufftResult_t rv;
    rv = cufftPlan2d(&plan, xpix, ypix, CUFFT_C2R);
    if (rv != CUFFT_SUCCESS) {
        char msg[1024];
        sprintf(msg, "Image::setup error planning FFT (%d)", rv);
        throw std::runtime_error(msg);
    }
    IFTIMER( timers["fft"] = new Timer(); )
}


void Image::operate(Array<cdata> &vis, Array<rdata> &img) {
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
    IFTIMER( timers["fft"]->start(); )
    rv = cufftExecC2R(plan, vis, img);
    IFTIMER( timers["fft"]->stop(); )
    if (rv != CUFFT_SUCCESS) {
        char msg[1024];
        sprintf(msg, "Image::operate error executing FFT (%d)", rv);
        throw std::runtime_error(msg);
    }
}

void Image::add_stat(std::string name) {
    // If the requested stat is already there, don't double-add it:
    for (unsigned i=0; i<stat_funcs.size(); i++) {
        if (name == stat_funcs[i]->name) return;
    }
    if (name=="max") {
        stat_funcs.push_back(new ImageMax(xpix,ypix,name));
    } else if (name=="pix") {
        stat_funcs.push_back(new ImageMaxPixel(xpix,ypix,name));
    } else if (name=="rms") {
        stat_funcs.push_back(new ImageRMS(xpix,ypix,name));
    } else if (name=="iqr") {
        stat_funcs.push_back(new ImageIQR(xpix,ypix,name));
    } else {
        char msg[1024];
        sprintf(msg, "Image::add_stat unknown stat type '%s'", name.c_str());
        throw std::runtime_error(msg);
    }
    _stat_offs.push_back(_stats.len()); // Offset into the stats results array
    _stats.resize(_stats.len() + stat_funcs.back()->num_stats());
    stat_funcs.back()->setup();
    IFTIMER( timers[name] = new Timer(); )
}

std::vector<std::string> Image::stat_names() const {
    std::vector<std::string> names;
    for (unsigned i=0; i<stat_funcs.size(); i++) {
        std::vector<std::string> p = stat_funcs[i]->provides();
        names.insert(names.end(), p.begin(), p.end());
    }
    return names;
}

std::vector<double> Image::stats(Array<rdata> &img) {
    for (unsigned i=0; i<stat_funcs.size(); i++) {
        IFTIMER( timers[stat_funcs[i]->name]->start(); )
        stat_funcs[i]->operate(img, _stats.d + _stat_offs[i]);
        IFTIMER( timers[stat_funcs[i]->name]->stop(); )
    }
    _stats.d2h();
    for (unsigned i=0; i<stat_funcs.size(); i++) 
        stat_funcs[i]->finalize(_stats.h + _stat_offs[i]);
    return std::vector<double>(_stats.h, _stats.h+_stats.len());
}

ImageStatistic::ImageStatistic(int _xpix, int _ypix, std::string _name) 
: tmp(false) 
{
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

void ImageMax::operate(Array<rdata> &img, double *result) {
    cub::DeviceReduce::Max(tmp.d, tmp_bytes, img.d, result, xpix*ypix);
}

void ImageMaxPixel::calc_buffer_size() {
    cub::DeviceReduce::ArgMax(NULL, tmp_bytes, (rdata *)NULL, 
            (cub::KeyValuePair<int, rdata> *)NULL, 
            xpix*ypix);
    tmp_bytes += sizeof(cub::KeyValuePair<int, rdata>);
}

__global__ void get_maxpix(cub::KeyValuePair<int,rdata> *argmax, double *result) {
    const int ii = blockDim.x*blockIdx.x + threadIdx.x;
    if (ii==0) {
        result[0] = (double)(argmax->value);
        result[1] = (double)(argmax->key);
    }
}

void ImageMaxPixel::operate(Array<rdata> &img, double *result) {
    size_t offs = sizeof(cub::KeyValuePair<int, rdata>);
    size_t tmp_bytes_only = tmp_bytes - offs;
    cub::KeyValuePair<int, rdata> *argmax = 
        (cub::KeyValuePair<int, rdata> *)(tmp.d + offs);
    cub::DeviceReduce::ArgMax(tmp.d, tmp_bytes_only, img.d, argmax, xpix*ypix);
    get_maxpix<<<1,1>>>(argmax, result);
}

void ImageMaxPixel::finalize(double *result) const {
    int maxpix = (int)(result[1]);
    result[1] = (maxpix / ypix);
    if (result[1] > xpix/2) { result[1] -= xpix; }
    result[2] = (maxpix % ypix);
    if (result[2] > ypix/2) { result[2] -= ypix; }
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

void ImageRMS::operate(Array<rdata> &img, double *result) {
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

void ImageIQR::operate(Array<rdata> &img, double *result) {
    size_t offs = sizeof(rdata)*xpix*ypix;
    size_t tmp_bytes_only = tmp_bytes - offs;
    rdata *sorted = (rdata *)(tmp.d + offs);
    cub::DeviceRadixSort::SortKeys(tmp.d, tmp_bytes_only,
            img.d, sorted, xpix*ypix);
    get_iqr<<<1,1>>>(sorted, result, xpix*ypix);
}
