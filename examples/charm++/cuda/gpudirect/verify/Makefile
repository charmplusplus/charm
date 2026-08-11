OPTS = -O0 -g

# Override on the command line to build against a different Charm++ build,
# e.g. make CHARM_DIR=../../../../../reconverse-linux-x86_64-cuda
CHARM_DIR ?= ../../../../../reconverse-linux-x86_64-amd
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

AMD_FLAGS = -D__HIP_PLATFORM_AMD__=ON

HAPICC = hipcc
HAPITOOLKIT_HOME ?= $(if $(ROCM_PATH),$(ROCM_PATH),/opt/rocm)
HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
HAPICC_LIB = -L$(HAPITOOLKIT_HOME)/lib
HAPI_FLAGS = -O3 -g -c -std=c++11 -use_fast_math
LD_LIBS =

TARGET = verify
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) $(AMD_FLAGS) -language charm++ -module CommonLBs -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $(AMD_FLAGS) $<

$(TARGET).def.h: $(TARGET).ci

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h
	$(CHARMC) $(AMD_FLAGS) -c $<

$(TARGET)CUDA.o: $(TARGET).cu
	$(HAPICC) $(AMD_FLAGS) -o $@ $(HAPI_FLAGS) $(HAPICC_INC) $(CHARM_INC) $(HAPICC_LIB) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
