CHARMDIR=../..
-include $(CHARMDIR)/include/conv-mach-opt.mak
CUDA_PATH=$(CUDA_DIR)
NVCC=$(CUDA_PATH)/bin/nvcc
NVCC_FLAGS=-c -use_fast_math -std=c++11 -DHAPI_MEMPOOL -DHAPI_TRACE # -device-debug -deviceemu
NVCC_INC=-I$(CUDA_PATH)/include -I$(CUDA_PATH)/extras/CUPTI/include -I..

all: libs
	cp libcudahybridapi.a $(CHARMDIR)/lib

libs: libcudahybridapi.a

install: libcudahybridapi.a
	cp libcudahybridapi.a $(CHARMDIR)/lib

libcudahybridapi.a: hybridapi.o
	-rm -f $@
	ar q $@ hybridapi.o

hybridapi.o: hapi_impl.cu hapi_impl.h hapi.h hapi_nvtx.h
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -o hybridapi.o hapi_impl.cu

clean:
	rm -f *.o *.a
