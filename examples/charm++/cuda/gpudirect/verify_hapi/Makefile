OPTS = -O0 -g

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include
CHARMC_FLAGS = -D__HIP_PLATFORM_AMD__=ON

HAPITOOLKIT_HOME ?= /opt/rocm
NVCC = hipcc
NVCC_FLAGS = -O3 -c -std=c++11 -use_fast_math
HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include

TARGET = verify
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)HAPI.o

$(TARGET): $(OBJS)
	$(CHARMC) $(CHARMC_FLAGS) -language charm++ -module CommonLBs -o $@ $(OBJS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $(CHARMC_FLAGS) $<

$(TARGET).def.h: $(TARGET).ci

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h
	$(CHARMC) $(CHARMC_FLAGS) -c $<

$(TARGET)HAPI.o: $(TARGET).cpp
	$(NVCC) $(CHARMC_FLAGS) -o $@ $(NVCC_FLAGS) $(HAPICC_INC) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
