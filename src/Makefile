# rfgpu Makfile

# The C++ compiler may need to be specified if the default on your system
# is not compatible with C++-11.  For example at NRAO, you should build using:
#
#  make CXX=/opt/local/compilers/gcc-5/bin/g++

# Also note the following paths may need to be changed for your system and/or
# install directory.  You can override these defaults on the command line using,
# for example:
#
#   make CUDA_DIR=/my/special/path/to/cuda
#   make install PREFIX=$HOME/local
#
CUDA_DIR ?= /opt/nvidia/cuda
PREFIX ?= /usr/local
PYTHON ?= python

### Things below this line should not need to be edited ###

all: rfgpu.so

clean: 
	rm -f *.o *.so

MISC_LIB = -lm

CUDA_INC = -I $(CUDA_DIR)/include -DUNIX -I ../extern/cub-1.8.0
CUDA_LIB = -L $(CUDA_DIR)/lib64 -lcufft -lcusparse -lnvToolsExt -lcuda -lcudart -lGL -lGLU $(MISC_LIB)

PY_INC = -I $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYBIND_INC = -I ../extern/pybind11-2.2.4/include

PYINSTPATH = $(shell $(PYTHON) -c "import sys; print('lib/python%d.%d/site-packages'%(sys.version_info.major,sys.version_info.minor))")

NVCC = nvcc -ccbin $(CXX) -arch sm_61

CFLAGS = -g -std=c++14

OBJS = grid.o image.o device.o
HDRS = array.h

# Uncomment these two lines to compile timing code
#CFLAGS += -DUSETIMER
#OBJS += timer.o

install: rfgpu.so
	mkdir -p $(PREFIX)/$(PYINSTPATH)
	cp -t $(PREFIX)/$(PYINSTPATH) $<

%.o : %.cu %.h $(HDRS) Makefile
	$(NVCC) $(CFLAGS) --compiler-options -fPIC \
		-c $(CUDA_INC) $< -o $@

rfgpu.so: rfgpu.C $(OBJS) $(HDRS) Makefile
	$(CXX) $(CFLAGS) -shared -fPIC \
		$(PY_INC) $(PYBIND_INC) $(CUDA_INC) \
		$< -o $@ $(OBJS) $(CUDA_LIB)
