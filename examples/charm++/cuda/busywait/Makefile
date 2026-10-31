-include ../../../common.mk
CHARMC = ../../../../bin/charmc $(OPTS)
OPTS = -DUSE_NVTX

# set CUDATOOLKIT_HOME to the CUDA toolkit directory
CUDATOOLKIT_HOME ?= /usr/local/cuda
NVCC = $(CUDATOOLKIT_HOME)/bin/nvcc
# GPU architecture the kernels are compiled for. Without this nvcc emits only
# its default target (sm_75 as of CUDA 13), and on a newer device the kernel
# can fail to launch with nothing but an unchecked error to show for it.
# sm_80 is the A100; override CUDA_ARCH for other devices (sm_86 A40, sm_90 H100).
CUDA_ARCH ?= sm_80
NVCC_FLAGS = -arch=$(CUDA_ARCH) -c -std=c++11
NVCC_INC = -I$(CUDATOOLKIT_HOME)/include
CHARM_INC = -I../../../../include

TARGET = busywait
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h
	$(CHARMC) -c $<

$(TARGET)CUDA.o: $(TARGET).cu
	$(NVCC) -o $@ $(NVCC_FLAGS) $(NVCC_INC) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun

test: all
	$(call run, ./$(TARGET) +p2)
