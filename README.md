# rfgpu

`rfgpu` is a CUDA-based library for GPU interferometric gridding and
imaging for the [realfast](http://realfast.io) project.  It is
incorporated into a more complete processing pipeline in
[rfpipe](https://github.com/realfastvla/rfpipe).

The low-level GPU routines are written entirely in CUDA/C++, and a
python interface is provided using pybind11.  This is compatible with
both python 2.7 and 3.x.

## Installation

Build and installation is based on a simple Makefile.  Two "header-only"
C++ libraries are required before `rfgpu` can be built.  These should be
downloaded in the `extern/` subdirectory via the provided download
script:

```
cd extern
./download.sh
```

After CUB and pybind11 are dowloaded, the library can be built and
installed in the `src` directory.  This will likely require
specifications of compiler and install path.  For example, at NRAO a
typical install into the realfast conda environment would use:

```
cd src
make CXX=/opt/local/compilers/gcc-5/bin/g++
make install PRFIX=/home/cbe-master/realfast/anaconda/envs/development3
```

At other sites, you may also need to specify the CUDA location via the
`CUDA_DIR` variable.

## Dependencies

- A C++-11 compatible C++ compiler (eg, GCC v4.8 or later)
- CUDA, including the cuFFT and cuSPARSE libraries
- CUB
- pybind11

