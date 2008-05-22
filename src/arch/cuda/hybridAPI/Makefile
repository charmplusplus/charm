CHARMDIR=~/charm
CHARMC=$(CHARMDIR)/bin/charmc

#LDFLAGS += -L/usr/local/cuda/lib -L~/NVIDIA_CUDA_SDK/lib
#LDLIBS += -lcuda -lcudart -lGL -lGLU -lcutil


NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -c -use_fast_math #-device-debug -deviceemu
NVCC_INC = -I/usr/local/cuda/include -I/~/NVIDIA_CUDA_SDK/common/inc

RM = rm -f

install: cudahybridapi.a
	cp cudahybridapi.a $(CHARMDIR)/lib

cudahybridapi.a: hybridapi.o wrqueue.o
	ar q cudahybridapi.a hybridapi.o wrqueue.o 

hybridapi.o: cuda-hybrid-api.h cuda-hybrid-api.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -o hybridapi.o cuda-hybrid-api.cu

wrqueue.o: wrqueue.cu wrqueue.h
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -o wrqueue.o wrqueue.cu 

