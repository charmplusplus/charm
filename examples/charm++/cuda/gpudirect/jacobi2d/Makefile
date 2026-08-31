OPTS = -O3

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

NVCC = nvcc
# GPU architecture the kernels are compiled for. Without this nvcc emits only
# its default target (sm_75 as of CUDA 13), and on a newer device the kernel
# can fail to launch with nothing but an unchecked error to show for it.
# sm_80 is the A100; override CUDA_ARCH for other devices (sm_86 A40, sm_90 H100).
CUDA_ARCH ?= sm_80
NVCC_FLAGS = -arch=$(CUDA_ARCH) -c -std=c++11 -use_fast_math $(OPTS)
LD_LIBS =

TARGET = jacobi2d
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci $(TARGET).h
	$(CHARMC) $<

$(TARGET).def.h: $(TARGET).ci $(TARGET).h

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h $(TARGET).h
	$(CHARMC) -c $<

$(TARGET)CUDA.o: $(TARGET).cu $(TARGET).h
	$(NVCC) -o $@ $(NVCC_FLAGS) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
