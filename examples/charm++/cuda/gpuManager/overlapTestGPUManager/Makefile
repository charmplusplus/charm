CHARMC=../../../../../net-linux-amd64-cuda/bin/charmc $(OPTS)

OBJS = overlapTest.o overlapTestCU.o

NVCC = /usr/local/cuda/bin/nvcc 
NVCC_FLAGS = -O3 -c -use_fast_math #-device-debug -deviceemu 
NVCC_INC = -I/usr/local/cuda/include -I/usr/local/NVIDIA_GPU_Computing_SDK/C/common/inc -I../../../../../src/arch/cuda/hybridAPI

#LD_LIBS += -lcuda -lcudart -lGL -lGLU -L../../../../NVIDIA_CUDA_SDK/lib -lcutil

LD_LIBS += -L/usr/local/NVIDIA_GPU_Computing_SDK/C/lib -lcutil_x86_64

export LD_RUN_PATH = /usr/local/cuda/lib64

all: overlapTest

overlapTest: $(OBJS)
	$(CHARMC) -language charm++ -o overlapTest $(OBJS) $(LD_LIBS) -tracemode projections

overlapTest.decl.h: overlapTest.ci
	$(CHARMC)  overlapTest.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o overlapTest charmrun

overlapTest.o: overlapTest.C overlapTest.decl.h
	$(CHARMC) -O3 -c overlapTest.C

overlapTestCU.o: overlapTest.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -o overlapTestCU.o overlapTest.cu
