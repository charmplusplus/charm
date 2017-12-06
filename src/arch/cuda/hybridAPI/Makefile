CHARMDIR=../..
-include $(CHARMDIR)/include/conv-mach-opt.mak
CUDA_PATH = $(CUDA_DIR)
NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS = -c -std=c++11 -use_fast_math -DGPU_MEMPOOL -D_MWAITXINTRIN_H_INCLUDED #-DGPU_PROFILE -DGPU_TRACE #-DGPU_MEMPOOL_DEBUG -DGPU_WRQ_VERBOSE #-device-debug -deviceemu
NVCC_INC = -I$(CUDA_PATH)/include -I..

all: libs
	cp libcudahybridapi.a $(CHARMDIR)/lib

libs: libcudahybridapi.a

install: libcudahybridapi.a
	cp libcudahybridapi.a $(CHARMDIR)/lib

libcudahybridapi.a: hybridapi.o wrqueue.o
	-rm -f $@
	ar q $@ hybridapi.o wrqueue.o 

hybridapi.o: hapi_src.cu hapi_src.h
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -o hybridapi.o hapi_src.cu

wrqueue.o: wrqueue.cu wrqueue.h wr.h 
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -o wrqueue.o wrqueue.cu

clean:
	rm -f *.o *.a 
