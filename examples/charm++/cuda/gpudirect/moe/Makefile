OPTS = -O3 -DHAPI_CUDA_CALLBACK

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

# cuBLAS ships beside the CUDA toolkit in the NVHPC tree (math_libs), not in
# the CUDA_DIR the runtime was built against. Derive it from the build's
# CUDA_DIR; override MATHLIBS on the make line for any other layout.
CUDA_DIR := $(shell sed -n 's/^CUDA_DIR="\(.*\)"/\1/p' $(CHARM_DIR)/include/conv-mach-opt.sh)
MATHLIBS ?= $(subst /cuda/,/math_libs/,$(CUDA_DIR))

NVCC = nvcc
NVCC_FLAGS = -c -std=c++17 -use_fast_math -gencode arch=compute_86,code=sm_86 $(OPTS) -I$(MATHLIBS)/include
LD_LIBS = -module EveryLB -module DiffusionLB -L$(MATHLIBS)/lib64 -lcublas -Wl,-rpath,$(MATHLIBS)/lib64

TARGET = moe
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
