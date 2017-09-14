-include ../../../common.mk
CHARMC = ../../../../bin/charmc $(OPTS)
OPTS = -DUSE_NVTX

# set CUDATOOLKIT_HOME to the CUDA toolkit directory
CUDATOOLKIT_HOME ?= /usr/local/cuda
NVCC = $(CUDATOOLKIT_HOME)/bin/nvcc
NVCC_FLAGS = -c -std=c++11
NVCC_INC = -I$(CUDATOOLKIT_HOME)/include
CHARM_INC = -I../../../../include
LD_LIBS = -lnvToolsExt

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
