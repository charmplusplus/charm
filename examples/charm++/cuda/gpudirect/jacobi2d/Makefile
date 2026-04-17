CHARM_DIR=/u/ajain18/reconverse_charm_integration/charm/multicore-linux-x86_64-cuda

CHARMC = $(CHARM_DIR)/bin/charmc $(USE_WR) $(OPTS)
CHARMC_FLAGS = 

# HAPITOOLKIT_HOME ?= /opt/rocm
HAPICC = nvcc
HAPICC_FLAGS = -O3 -g -c -std=c++11 -use_fast_math
# HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
# HAPICC_LIB = -L$(HAPITOOLKIT_HOME)/lib
CHARM_INC = -I$(CHARM_DIR)/include
LD_LIBS =

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
