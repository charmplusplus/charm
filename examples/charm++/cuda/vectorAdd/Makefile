CHARMC=../../../../bin/charmc $(OPTS)

OBJS = vectorAdd.o vectorAddCU.o

CUDA_LEVEL=35
NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -O3 -c -use_fast_math -DGPU_MEMPOOL #-device-debug -deviceemu
NVCC_FLAGS += -arch=compute_$(CUDA_LEVEL) -code=sm_$(CUDA_LEVEL)
NVCC_INC = -I/usr/local/cuda/include  -I../../../../../src/arch/cuda/hybridAPI
CHARMINC = -I../../../../include
LD_LIBS= -lcublas
all: vectorAdd

vectorAdd: $(OBJS)
	$(CHARMC) -language charm++ -o vectorAdd $(OBJS) $(LD_LIBS)

vectorAdd.decl.h: vectorAdd.ci
	$(CHARMC)  vectorAdd.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o vectorAdd charmrun

vectorAdd.o: vectorAdd.C vectorAdd.decl.h
	$(CHARMC) -O3 -c vectorAdd.C

vectorAddCU.o: vectorAdd.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $(CHARMINC) -o vectorAddCU.o vectorAdd.cu

test: all
	./charmrun ./vectorAdd +p2 2 8  #++local ++verbose
