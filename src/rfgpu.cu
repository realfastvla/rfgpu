#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cusparse.h>

#include "rfgpu.h"

using namespace rfgpu;

// TODO raise exception
#define check_rv(func) \
    if (rv!=CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "%s returned %d\n", func, rv); \
    }

Grid::Grid(int _nbl, int _nchan, int _ntime, int _upix, int _vpix) {
    nbl = _nbl;
    nchan = _nchan;
    ntime = _ntime;
    upix = _upix;
    vpix = _vpix;

    h_one = make_float2(1.0,0.0);
    h_zero = make_float2(0.0,0.0);

    // TODO create cusparse context
    cusparseCreateMatDescr(&descr);
    cell = 80.0; // 80 wavelengths == ~42' FoV

    allocate();
}

void Grid::allocate() {
    u.resize(nbl);
    v.resize(nbl);
    freq.resize(nchan);

    G_vals.resize(ncol());
    G_rows.resize(nrow()+1);
    G_cols.resize(ncol());
    G_cols0.resize(ncol());
    G_chan.resize(ncol());

    shift.resize(nchan);
}

void Grid::set_uv(float *_u, float *_v) {
    for (int i=0; i<nbl; i++) {
        u[i] = _u[i];
        v[i] = _v[i];
    }
}

void Grid::set_freq(float *_freq) {
    for (int i=0; i<nchan; i++) { freq[i] = _freq[i]; }
}

void Grid::set_shift(int *_shift) {
    maxshift=0;
    for (int i=0; i<nchan; i++) {
        if (_shift[i]>maxshift) { maxshift=_shift[i]; }
    }
    if (maxshift>ntime) { } // TODO raise error
    cudaMemcpy(shift.d, _shift, shift.size(), cudaMemcpyHostToDevice);
}

void Grid::compute() {

    //printf("nrow=%d ncol=%d\n", nrow(), ncol());

    // compute grid pix location for each input vis
    nnz = 0;
    for (int ibl=0; ibl<nbl; ibl++) {
        for (int ichan=0; ichan<nchan; ichan++) {
            int x = round((u[ibl]*freq[ichan])/cell);
            int y = round((v[ibl]*freq[ichan])/cell); 
            if (y<0) { y*=-1; x*=-1; } // TODO need to conjugate data too...
            if (x>upix/2) { x += upix; }
            if (x<0) { x += upix; }
            if (x<upix && x>=0 && y<vpix && y>=0) {
                G_pix.h[nnz] = y*upix + x;
                G_cols0.h[nnz] = ibl*nchan + ichan;
                nnz++;
            } 
        }
    }
    G_pix.h2d();
    G_cols0.h2d();

    cusparseStatus_t rv;

    // on GPU, sort and compress into CSR matrix format
    size_t pbuf_size;
    rv = cusparseXcoosort_bufferSizeExt(sparse, nrow(), ncol(), nnz, 
            G_pix.d, G_cols.d, &pbuf_size);
    check_rv("cusparseXcoosort_bufferSizeExt");

    Array<char> pbuf(pbuf_size);
    Array<int> perm(nnz);
    rv = cusparseCreateIdentityPermutation(sparse, nnz, perm.d);
    check_rv("cusparseCreateIdentityPermutation");

    rv = cusparseXcoosortByRow(sparse, nrow(), ncol(), nnz,
            G_pix.d, G_cols0.d, perm.d, (void *)pbuf.d);
    check_rv("cusparseXcoosortByRow");

    rv = cusparseXcoo2csr(sparse, G_pix.d, nnz, nrow(), G_rows.d,
            CUSPARSE_INDEX_BASE_ZERO);
    check_rv("cusparseXcoo2csr");

    // Fill in normalization factors (number of vis per grid point)
    G_rows.d2h();
    for (int i=0; i<nrow(); i++) {
        for (int j=G_rows.h[i]; j<G_rows.h[i+1]; j++) {
            G_vals.h[j].x = 1.0/((float)G_rows.h[i+1] - (float)G_rows.h[i]);
            G_vals.h[j].y = 0.0;
        }
    }
    G_vals.h2d();

    // retrieve channel idx of each data point
    G_cols.d2h();
    for (int i=0; i<nnz; i++) { G_chan.h[i] = G_cols.h[i] % nchan; }
    G_chan.h2d();
}

__global__ void adjust_cols(int *ocol, int *icol, int *chan,
        int *shift, int itime, int nchan, int nnz, int ntime) {
    const int ii = blockDim.x*blockIdx.x + threadIdx.x;
    __shared__ int lshift[2048]; // max nchan=2048
    for (int i=threadIdx.x; i<nchan; i+=blockDim.x) {
        lshift[i] = shift[i];
    }
    __syncthreads();
    if (ii<nnz) { ocol[ii] = icol[ii]*ntime + lshift[chan[ii]] + itime; }
}

void Grid::operate(cdata *in, cdata *out, int itime) {
    cusparseStatus_t rv;
    adjust_cols<<<nbl, nchan>>>(G_cols.d, G_cols0.d, G_chan.d, shift.d, 
            itime, nchan, nnz, ntime);

    rv = cusparseCcsrmv(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            nrow(), ncol()*ntime, nnz, &h_one, descr,
            G_vals.d, G_rows.d, G_cols.d,
            in, &h_zero, out);
    check_rv("cusparseCcsrmv");
}

Image::Image() {
    plan = NULL; 
    xpix = ypix = 0;
}

Image::~Image() {
    if (plan) cufftDestroy(plan);
}

void Image::setup() {
    cufftPlan2d(&plan, xpix, ypix, CUFFT_C2R); // TODO check for error
}

void Image::operate(cufftComplex *vis, cufftReal *img) {
    cufftExecC2R(plan, vis, img);
}

