
all: rfgpu.so

clean: 
	rm -f *.o *.so

MISC_LIB = -lm

CUDA_DIR = /opt/nvidia/cuda
CUDA_INC = -I $(CUDA_DIR)/include -DUNIX
CUDA_LIB = -L $(CUDA_DIR)/lib64 -lcufft -lcusparse -lnvToolsExt -lcuda -lcudart -lGL -lGLU $(MISC_LIB)

PY_INC = -I /usr/local/stow/python-2.7.10/include/python2.7
PYBIND_INC = -I /users/pdemores/src/pybind11-2.2.1/include

CXX = /opt/local/compilers/gcc-5/bin/g++
NVCC = nvcc -g -ccbin $(CXX) -arch sm_61

OBJS = grid.o image.o

%.o : %.cu %.h
	$(NVCC) --compiler-options -fPIC \
		-c $(CUDA_INC) $< -o $@

rfgpu.so: rfgpu.C $(OBJS)
	$(CXX) -shared -static-libstdc++ -std=c++11 -fPIC \
		$(PY_INC) $(PYBIND_INC) $(CUDA_INC) \
		$< -o $@ $(OBJS) $(CUDA_LIB)
