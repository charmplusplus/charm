-include ../../../common.mk
# Ensure that CUDA_DIR, NVIDIA_CUDA_SDK, and LD_LIBRARY_PATH are set
# in your environment. For the tests to work right, those settings
# need to be in a login script.

CUDATOOLKIT_HOME ?= /usr/local/cuda
NVCC=$(CUDATOOLKIT_HOME)/bin/nvcc
CHARMDIR=../../../..
AMPICC=$(CHARMDIR)/bin/ampicc
AMPICXX=$(CHARMDIR)/bin/ampicxx

DEFS=#-DUSE_WR

CXX=$(AMPICXX)
LD=$(AMPICXX)
CXXFLAGS=$(DEFS)

simple: simple-cuda.o

simple-cuda.o: simple-cuda.cu
	$(NVCC) $(DEFS) -std=c++11 -I$(CHARMDIR)/include -c simple-cuda.cu

test:
	./simple +vp 1
	./simple +vp 2
	./simple +vp 16
	$(call run, +p2 ./simple +vp 16)

clean:
	rm -f charmrun simple *.o ampirun
