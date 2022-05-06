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

#include "timer.h"
#include "grid.h"
#include "device.h"

using namespace rfgpu;

#define cusparse_check_rv(func) \
    if (rv!=CUSPARSE_STATUS_SUCCESS) { \
        char msg[1024]; \
        sprintf(msg, "cusparse error: %s returned %d", func, rv); \
        throw std::runtime_error(msg); \
    }

#define array_dim_check(func,array,expected) { \
    if (array.dims() != expected) { \
        char msg[1024]; \
        sprintf(msg, "%s: array dimension error", func); \
        throw std::invalid_argument(msg); \
    } \
    if (array.has_device(_device)==false) { \
        char msg[1024]; \
        sprintf(msg, "%s: array not defined on device %d", func, _device); \
        throw std::invalid_argument(msg); \
    }\
}

Grid::Grid(int _nbl, int _nchan, int _ntime, int _upix, int _vpix, int device) 
: G_cols(false), shift(false), mvbuf(false), OnDevice(device)  // Any GPU-only arrays go here
{
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
    //rv = cusparseCreateMatDescr(&descr);
    //cusparse_check_rv("cusparseCreateMatDescr");
    descr = NULL;

    cell = 80.0; // 80 wavelengths == ~42' FoV

    maxshift = 0;

#ifdef USETIMER
    timers["grid"] = new Timer();
    timers["cols"] = new Timer();
    timers["ds"] = new Timer();
    timers["compute"] = new Timer();
    timers["conj"] = new Timer();
#endif

    allocate();
    reset_device();
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
    int _maxshift=0;
    for (int i=0; i<nchan; i++) {
        if (_shift[i]>_maxshift) { _maxshift=_shift[i]; }
        if (_shift[i]<0) {
            char msg[1024];
            sprintf(msg, "Grid::set_shift negative shift not allowed "
                    "(ichan=%d shift=%d)", i, _shift[i]);
            throw std::invalid_argument(msg);
        }
    }
    if (_maxshift>ntime) { 
        char msg[1024];
        sprintf(msg, 
                "Grid::set_shift max shift out of range (maxshift=%d ntime=%d)",
                _maxshift, ntime);
        throw std::invalid_argument(msg);
    }
    maxshift = _maxshift;
    CheckDevice cd(this);
    cudaMemcpy(shift.d, _shift.data(), shift.size(), cudaMemcpyHostToDevice);
}

void Grid::compute() {
    CheckDevice cd(this);

    //printf("nrow=%d ncol=%d\n", nrow(), ncol());
    IFTIMER( timers["compute"]->start(); )

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
            G_pix.d, G_cols0.d, &pbuf_size);
    cusparse_check_rv("cusparseXcoosort_bufferSizeExt");

    Array<char> pbuf(pbuf_size,false);
    Array<int> perm(nnz,false);
    rv = cusparseCreateIdentityPermutation(sparse, nnz, perm.d);
    cusparse_check_rv("cusparseCreateIdentityPermutation");

    rv = cusparseXcoosortByRow(sparse, nrow(), ncol(), nnz,
            G_pix.d, G_cols0.d, perm.d, (void *)pbuf.d);
    cusparse_check_rv("cusparseXcoosortByRow");

    rv = cusparseXcoo2csr(sparse, G_pix.d, nnz, nrow(), G_rows.d,
            CUSPARSE_INDEX_BASE_ZERO);
    cusparse_check_rv("cusparseXcoo2csr");

    // Fill in normalization factors
    G_rows.d2h();
    for (int i=0; i<nrow(); i++) {
        for (int j=G_rows.h[i]; j<G_rows.h[i+1]; j++) {
            // This is something like uniform weighting:
            //G_vals.h[j].x = 1.0/((float)G_rows.h[i+1] - (float)G_rows.h[i]);
            // This is natural weighting:
            G_vals.h[j].x = 1.0/(2.0*nnz);
            G_vals.h[j].y = 0.0;
        }
    }
    G_vals.h2d();

    // Create matrix descriptor
    if (descr) {
        rv = cusparseDestroySpMat(descr);
        cusparse_check_rv("cusparseDestroySpMat");
    }
    rv = cusparseCreateCsr(&descr, nrow(), ncol()*ntime, nnz, 
            G_rows.d, G_cols.d, G_vals.d, 
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_C_32F);
    cusparse_check_rv("cusparseCreateCsr");

    // retrieve channel idx of each data point
    G_cols0.d2h();
    for (int i=0; i<nnz; i++) { G_chan.h[i] = G_cols0.h[i] % nchan; }
    G_chan.h2d();
    IFTIMER( timers["compute"]->stop(); )
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

void Grid::conjugate(Array<cdata> &data) {
    array_dim_check("Grid::conjugate", data, indim());
    CheckDevice cd(this);
    IFTIMER( timers["conj"]->start(); )
    conjugate_data<<<nbl,512>>>(data.dd[_device], conj.d, nchan, ntime);
    IFTIMER( timers["conj"]->stop(); )
}

// Call with nbl thread blocks
// TODO there may be problems if ntime is not divisible by 2
__global__ void downsample_data(cdata *dat, int nchan, int ntime) {
    const int ibl = blockIdx.x;
    const int offs = ibl*nchan*ntime;
    for (int ichan=0; ichan<nchan; ichan++) {
        for (int itime=2*threadIdx.x; itime<ntime; itime+=2*blockDim.x) {
            const int ii = offs + ichan*ntime + itime;
            float2 x0 = dat[ii];
            float2 x1 = dat[ii+1];
            const int oo = offs + ichan*ntime + itime/2;
            __syncthreads();
            dat[oo].x = 0.5*(x0.x + x1.x);
            dat[oo].y = 0.5*(x0.y + x1.y);
        }
    }
}

void Grid::downsample(Array<cdata> &data) {
    array_dim_check("Grid::downsample", data, indim());
    CheckDevice cd(this);
    IFTIMER( timers["ds"]->start(); )
    downsample_data<<<nbl,512>>>(data.dd[_device], nchan, ntime);
    IFTIMER( timers["ds"]->stop(); )
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

void Grid::operate(Array<cdata> &in, Array<cdata> &out, int itime) {
    array_dim_check("Grid::operate(in)", in, indim());
    array_dim_check("Grid::operate(out)", out, outdim());
    operate(in.dd[_device], out.dd[_device], itime);
}

void Grid::operate(cdata *in, cdata *out, int itime) {
    if ((itime+maxshift)>=ntime) {
        char msg[1024];
        sprintf(msg, 
                "Grid::operate itime(%d)+maxshift(%d) >= ntime(%d)", 
                itime, maxshift, ntime);
        throw std::invalid_argument(msg);
    }

    CheckDevice cd(this);

    // wrap in and out as cusparseDnVec
    cusparseStatus_t rv;
    cusparseDnVecDescr_t v_in, v_out;
    rv = cusparseCreateDnVec(&v_in, ncol()*ntime, in, CUDA_C_32F);
    cusparse_check_rv("cusparseCreateDnVec(in)");
    rv = cusparseCreateDnVec(&v_out, nrow(), out, CUDA_C_32F);
    cusparse_check_rv("cusparseCreateDnVec(out)");

    // Check and allocate working space
    size_t bufsize;
    rv = cusparseSpMV_bufferSize(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one, descr, v_in, &h_zero, v_out, CUDA_C_32F,
            CUSPARSE_SPMV_CSR_ALG1, &bufsize);
    cusparse_check_rv("cusparseSpMV_bufferSize");
    if (mvbuf.size() != bufsize) { mvbuf.resize(bufsize); }

    // Need to keep n threads per block less than 1024
    // Can we automatically get max thread per block?
    int nthread = 512;
    int nblock = (nbl*nchan)/nthread;
    if ((nbl*nchan)%nthread) { nblock++; }

    IFTIMER( timers["cols"]->start(); )
    adjust_cols<<<nblock, nthread>>>(G_cols.d, G_cols0.d, G_chan.d, shift.d, 
            itime, nchan, nnz, ntime);
    IFTIMER( timers["cols"]->stop(); )

    IFTIMER( timers["grid"]->start(); )
#if 0 
    rv = cusparseCcsrmv(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            nrow(), ncol()*ntime, nnz, &h_one, descr,
            G_vals.d, G_rows.d, G_cols.d,
            in, &h_zero, out);
#endif
    rv = cusparseSpMV(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one, descr, v_in, &h_zero, v_out, CUDA_C_32F,
            CUSPARSE_SPMV_CSR_ALG1, mvbuf.d);
    IFTIMER (timers["grid"]->stop(); )
    cusparse_check_rv("cusparseCcsrmv");

    // Clean up
    rv = cusparseDestroyDnVec(v_in);
    cusparse_check_rv("cusparseDestroyDnVec(in)");
    rv = cusparseDestroyDnVec(v_out);
    cusparse_check_rv("cusparseDestroyDnVec(out)");
}

