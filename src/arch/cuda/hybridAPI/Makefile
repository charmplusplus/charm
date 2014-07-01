#CUDA_DIR=/usr/local/cuda
CHARMDIR=../..
CHARMC=$(CHARMDIR)/bin/charmc
CUDA_PATH=$(shell CHARMINC=../.; . $$CHARMINC/conv-mach-opt.sh; echo $$CUDA_DIR;  )
NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS = -c -use_fast_math -DGPU_MEMPOOL #-DGPU_PROFILE -DGPU_TRACE #-DGPU_MEMPOOL_DEBUG -DGPU_WRQ_VERBOSE #-device-debug -deviceemu
NVCC_INC = -I$(CUDA_PATH)/include -I..

all: libs
	cp libcudahybridapi.a $(CHARMDIR)/lib

libs: libcudahybridapi.a

install: libcudahybridapi.a
	cp libcudahybridapi.a $(CHARMDIR)/lib
#	cp hybridapi.o wrqueue.o $(CHARMDIR)/tmp

libcudahybridapi.a: hybridapi.o wrqueue.o
	-rm -f $@
	ar q $@ hybridapi.o wrqueue.o 

hybridapi.o: cuda-hybrid-api.cu cuda-hybrid-api.h
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -o hybridapi.o cuda-hybrid-api.cu

wrqueue.o: wrqueue.cu wrqueue.h wr.h 
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -o wrqueue.o wrqueue.cu

clean:
	rm -f *.o *.a 
