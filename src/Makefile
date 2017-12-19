
all:

clean: 
	rm -f *.o gridtest

MISC_LIB = -lm

CUDA_DIR = /opt/nvidia/cuda
CUDA_INC = -I $(CUDA_DIR)/include -DUNIX
CUDA_LIB = -L $(CUDA_DIR)/lib64 -lcufft -lcusparse -lnvToolsExt -lcuda -lcudart -lGL -lGLU $(MISC_LIB)

#NVCC = nvcc -ccbin $(CUDA_DIR)/gcc
NVCC = nvcc -g -ccbin /opt/local/compilers/gcc-5/bin/gcc

%.o : %.cu %.h
	$(NVCC) -c $(CUDA_INC) $< -o $@
