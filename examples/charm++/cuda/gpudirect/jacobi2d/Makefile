OPTS = -O3 -DHAPI_CUDA_CALLBACK

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include
CHARMC_FLAGS = -D__HIP_PLATFORM_AMD__=ON

HAPITOOLKIT_HOME ?= /opt/rocm
HAPICC = nvcc
HAPICC_FLAGS = -c -std=c++11
HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
HAPICC_LIB = -L$(HAPITOOLKIT_HOME)/lib

LD_LIBS = -module EveryLB



TARGET = jacobi2d
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)HAPI.o

$(TARGET): $(OBJS)
	$(CHARMC) $(CHARMC_FLAGS) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $(CHARMC_FLAGS) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h
	$(CHARMC) $(CHARMC_FLAGS) -c $<

$(TARGET)HAPI.o: $(TARGET).cu
	$(HAPICC) $(CHARMC_FLAGS) -o $@ $(HAPICC_FLAGS) $(HAPICC_INC) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun

test: all
	$(call run, ./$(TARGET) +p2)
