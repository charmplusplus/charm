-include ../../../common.mk
# Ensure that CUDA_DIR, NVIDIA_CUDA_SDK, and LD_LIBRARY_PATH are set
# in your environment. For the tests to work right, those settings
# need to be in a login script.

CUDATOOLKIT_HOME ?= /usr/local/cuda
NVCC=$(CUDATOOLKIT_HOME)/bin/nvcc
# GPU architecture the kernels are compiled for. Without this nvcc emits only
# its default target (sm_75 as of CUDA 13), and on a newer device the kernel
# can fail to launch with nothing but an unchecked error to show for it.
# sm_80 is the A100; override CUDA_ARCH for other devices (sm_86 A40, sm_90 H100).
CUDA_ARCH ?= sm_80
NVCC_INC = -I../../../../include
CHARMC_INC = -I$(CUDATOOLKIT_HOME)/include
LD_LIBS = -L$(CUDATOOLKIT_HOME)/lib64 -lcudart
CHARMDIR=../../../..
AMPICC=$(CHARMDIR)/bin/ampicc
AMPICXX=$(CHARMDIR)/bin/ampicxx

DEFS=#-DUSE_WR

CXX=$(AMPICXX)
LD=$(AMPICXX)
CXXFLAGS=$(DEFS)

simple: simple-cuda.o simple.o
	$(AMPICXX) -o $@ $^ $(LD_LIBS)

simple-cuda.o: simple-cuda.cu
	$(NVCC) -arch=$(CUDA_ARCH) $(NVCC_INC) $(DEFS) -std=c++11 -c $<

simple.o: simple.C
		$(AMPICXX) $(CHARMC_INC) -c $<


test:
	./simple +vp 1
	./simple +vp 2
	./simple +vp 16
	$(call run, +p2 ./simple +vp 16)

clean:
	rm -f charmrun simple *.o ampirun
