-include ../../../common.mk
CHARM_DIR ?= ../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)

# set CUDATOOLKIT_HOME to the CUDA toolkit directory
CUDATOOLKIT_HOME ?= /usr/local/cuda
NVCC = $(CUDATOOLKIT_HOME)/bin/nvcc
CUDA_ARCH ?= sm_80
NVCC_FLAGS = -c -std=c++14 -arch=$(CUDA_ARCH)
NVCC_INC = -I$(CUDATOOLKIT_HOME)/include
CHARM_INC = -I../../../../include

TARGET = hapi_verify
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h
	$(CHARMC) -c $<

$(TARGET)CUDA.o: $(TARGET).cu
	$(NVCC) -o $@ $(NVCC_FLAGS) $(NVCC_INC) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun

# Needs a GPU at runtime (see the GPU-series validation protocol in
# doc/reconverse-merge-plan.md); deliberately not in the CPU CI tier.
test: all
	$(call run, ./$(TARGET) +p2)
