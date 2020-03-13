-include ../../../../common.mk
CHARMC = ../../../../../bin/charmc $(OPTS)
OPTS = -O3

# Set CUDATOOLKIT_HOME to the CUDA toolkit directory
CUDATOOLKIT_HOME ?= $(CUDA_HOME)
NVCC = $(CUDATOOLKIT_HOME)/bin/nvcc
NVCC_FLAGS = -O3 -c -std=c++11 -use_fast_math
NVCC_INC = -I$(CUDATOOLKIT_HOME)/include
CHARM_INC = -I../../../../../include
LD_LIBS =

TARGET = jacobi2d
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
	$(call run, ./$(TARGET) +p2 32 16 )
