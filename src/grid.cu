#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <vector>
#include <algorithm>
#include <stdexcept>

#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cusparse.h>

#include "grid.h"

using namespace rfgpu;

// TODO raise exception
#define cusparse_check_rv(func) \
    if (rv!=CUSPARSE_STATUS_SUCCESS) { \
        char msg[1024]; \
        sprintf(msg, "cusparse error: %s returned %d", func, rv); \
        throw std::runtime_error(msg); \
    }

Grid::Grid(int _nbl, int _nchan, int _ntime, int _upix, int _vpix) {
    nbl = _nbl;
    nchan = _nchan;
    ntime = _ntime;
    upix = _upix;
    vpix = _vpix;

    h_one = make_float2(1.0,0.0);
    h_zero = make_float2(0.0,0.0);

    cusparseStatus_t rv;
    rv = cusparseCreate(&sparse);
    cusparse_check_rv("cusparseCreate");
    rv = cusparseCreateMatDescr(&descr);
    cusparse_check_rv("cusparseCreateMatDescr");

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
    G_pix.resize(ncol());

    shift.resize(nchan);
    conj.resize(nbl);
}

void Grid::set_uv(const std::vector<float> &_u, const std::vector<float> &_v) {
    if (_u.size()!=nbl || _v.size()!=nbl) {
        char msg[1024];
        sprintf(msg, "Grid::set_uv array size error (u=%d v=%d nbl=%d)",
                _u.size(), _v.size(), nbl);
        throw std::invalid_argument(msg);
    }
    for (int i=0; i<nbl; i++) {
        u[i] = _u[i];
        v[i] = _v[i];
    }
}

void Grid::set_freq(const std::vector<float> &_freq) {
    if (_freq.size()!=nchan) {
        char msg[1024];
        sprintf(msg, "Grid::set_freq array size error (freq=%d nchan=%d)",
                _freq.size(), nchan);
        throw std::invalid_argument(msg);
    }
    for (int i=0; i<nchan; i++) { freq[i] = _freq[i]; }
}

void Grid::set_shift(const std::vector<int> &_shift) {
    if (_shift.size()!=nchan) {
        char msg[1024];
        sprintf(msg, "Grid::set_shift array size error (shift=%d nchan=%d)",
                _shift.size(), nchan);
        throw std::invalid_argument(msg);
    }
    maxshift=0;
    for (int i=0; i<nchan; i++) {
        if (_shift[i]>maxshift) { maxshift=_shift[i]; }
    }
    if (maxshift>ntime) { 
        char msg[1024];
        sprintf(msg, 
                "Grid::set_shift max shift out of range (maxshift=%d ntime=%d)",
                maxshift, ntime);
        throw std::invalid_argument(msg);
    }
    cudaMemcpy(shift.d, _shift.data(), shift.size(), cudaMemcpyHostToDevice);
}

void Grid::compute() {

    //printf("nrow=%d ncol=%d\n", nrow(), ncol());

    // compute grid pix location for each input vis
    nnz = 0;
    for (int ibl=0; ibl<nbl; ibl++) {
        for (int ichan=0; ichan<nchan; ichan++) {
            int x = round((u[ibl]*freq[ichan])/cell);
            int y = round((v[ibl]*freq[ichan])/cell); 
            if (y<0) { y*=-1; x*=-1; conj.h[ibl]=1; }
            else { conj.h[ibl]=0; }
            if (x<=upix/2 && x>=-upix/2 && y<vpix && y>=0) {
                if (x<0) x += upix;
                G_pix.h[nnz] = x*vpix + y;
                G_cols0.h[nnz] = ibl*nchan + ichan;
                nnz++;
            } 
        }
    }
    G_pix.h2d();
    G_cols0.h2d();
    conj.h2d();

    cusparseStatus_t rv;

    // on GPU, sort and compress into CSR matrix format
    size_t pbuf_size;
    rv = cusparseXcoosort_bufferSizeExt(sparse, nrow(), ncol(), nnz, 
            G_pix.d, G_cols.d, &pbuf_size);
    cusparse_check_rv("cusparseXcoosort_bufferSizeExt");

    Array<char> pbuf(pbuf_size);
    Array<int> perm(nnz);
    rv = cusparseCreateIdentityPermutation(sparse, nnz, perm.d);
    cusparse_check_rv("cusparseCreateIdentityPermutation");

    rv = cusparseXcoosortByRow(sparse, nrow(), ncol(), nnz,
            G_pix.d, G_cols0.d, perm.d, (void *)pbuf.d);
    cusparse_check_rv("cusparseXcoosortByRow");

    rv = cusparseXcoo2csr(sparse, G_pix.d, nnz, nrow(), G_rows.d,
            CUSPARSE_INDEX_BASE_ZERO);
    cusparse_check_rv("cusparseXcoo2csr");

    // Fill in normalization factors (number of vis per grid point)
    // TODO maybe these should just be set to 1.0?
    G_rows.d2h();
    for (int i=0; i<nrow(); i++) {
        for (int j=G_rows.h[i]; j<G_rows.h[i+1]; j++) {
            G_vals.h[j].x = 1.0/((float)G_rows.h[i+1] - (float)G_rows.h[i]);
            G_vals.h[j].y = 0.0;
        }
    }
    G_vals.h2d();

    // retrieve channel idx of each data point
    G_cols0.d2h();
    for (int i=0; i<nnz; i++) { G_chan.h[i] = G_cols0.h[i] % nchan; }
    G_chan.h2d();
}

// Call with nbl thread blocks
__global__ void conjugate_data(cdata *dat, int *conj, int nchan, int ntime) {
    const int ibl = blockIdx.x;
    const int offs = ibl*nchan*ntime;
    if (conj[ibl]) { 
        for (int i=threadIdx.x; i<nchan*ntime; i+=blockDim.x) {
            dat[offs+i].y *= -1.0;
        }
    }
}

void Grid::conjugate(Array<cdata,true> &data) {
    conjugate_data<<<nbl,512>>>(data.d, conj.d, nchan, ntime);
}

__global__ void adjust_cols(int *ocol, int *icol, int *chan,
        int *shift, int itime, int nchan, int nnz, int ntime) {
    const int ii = blockDim.x*blockIdx.x + threadIdx.x;
    __shared__ int lshift[2048]; // max nchan=2048 TODO 
    for (int i=threadIdx.x; i<nchan; i+=blockDim.x) {
        lshift[i] = shift[i];
    }
    __syncthreads();
    if (ii<nnz) { ocol[ii] = icol[ii]*ntime + lshift[chan[ii]] + itime; }
}

void Grid::operate(Array<cdata,true> &in, Array<cdata,true> &out, int itime) {
    if (in.len()!=nbl*nchan*ntime) {
        char msg[1024];
        sprintf(msg, "Grid::operate input array size (%d) != expected (%d)",
                in.len(), nbl*nchan*ntime);
        throw std::invalid_argument(msg);
    }
    if (out.len()!=upix*vpix) {
        char msg[1024];
        sprintf(msg, "Grid::operate output array size (%d) != expected (%d)",
                in.len(), upix*vpix);
        throw std::invalid_argument(msg);
    }
    operate(in.d, out.d, itime);
}

void Grid::operate(cdata *in, cdata *out, int itime) {
    if (itime>=ntime) {
        char msg[1024];
        sprintf(msg, "Grid::operate itime (%d) >= ntime (%d)", itime, ntime);
        throw std::invalid_argument(msg);
    }

    adjust_cols<<<nbl, nchan>>>(G_cols.d, G_cols0.d, G_chan.d, shift.d, 
            itime, nchan, nnz, ntime);

    cusparseStatus_t rv;
    rv = cusparseCcsrmv(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            nrow(), ncol()*ntime, nnz, &h_one, descr,
            G_vals.d, G_rows.d, G_cols.d,
            in, &h_zero, out);
    cusparse_check_rv("cusparseCcsrmv");
}

