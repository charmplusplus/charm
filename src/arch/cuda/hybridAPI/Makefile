CHARMDIR=../../
CHARMC=$(CHARMDIR)/bin/charmc

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -c -use_fast_math #-DGPU_PROFILE -DGPU_TRACE #-DGPU_PRINT_BUFFER_ALLOCATE #-device-debug -deviceemu
NVCC_INC = -I/usr/local/cuda/include -I/usr/local/NVIDIA_CUDA_SDK/common/inc -I../

RM = rm -f

all: libs
	cp libcudahybridapi.a $(CHARMDIR)/lib

libs: libcudahybridapi.a

install: libcudahybridapi.a
	cp libcudahybridapi.a $(CHARMDIR)/lib
#	cp hybridapi.o wrqueue.o $(CHARMDIR)/tmp

libcudahybridapi.a: hybridapi.o wrqueue.o
	-rm $@
	ar q $@ hybridapi.o wrqueue.o 

hybridapi.o: cuda-hybrid-api.cu cuda-hybrid-api.h
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -o hybridapi.o cuda-hybrid-api.cu

wrqueue.o: wrqueue.cu wrqueue.h wr.h 
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -o wrqueue.o wrqueue.cu 

clean:
	rm -f *.o *.a 
