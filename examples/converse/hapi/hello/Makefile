-include ../../../common.mk
CHARMC = ../../../../bin/charmc $(OPTS)

CUDA_DIR ?= $(CUDA_HOME)
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -c -std=c++11
CHARM_INC = ../../../../include
CHARM_LIB = ../../../../lib
LD_LIBS = -L$(CUDA_DIR)/lib64 -lcudart -L$(CHARM_LIB) -lhapi

TARGET = hello
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h
	$(CHARMC) -c $< -I$(CUDA_DIR)/include

$(TARGET)CUDA.o: $(TARGET).cu
	$(NVCC) -o $@ $(NVCC_FLAGS) -I$(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o charmrun $(TARGET)

test: all
	$(call run, ./$(TARGET) +p2)
