-include ../../../../common.mk

# Defaults to CUDA, matching the other examples under examples/charm++/cuda.
# For a ROCm build, override the toolchain on the command line:
#
#   make HAPICC=hipcc HAPITOOLKIT_HOME=/opt/rocm \
#        CHARMC_FLAGS=-D__HIP_PLATFORM_AMD__=ON
#
# CHARM_DIR is taken from the build tree this Makefile sits in; override it
# when building against a Charm++ installed elsewhere.
OPTS = -O3 -DHAPI_CUDA_CALLBACK

CHARM_DIR ?= ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include
CHARMC_FLAGS =

HAPITOOLKIT_HOME ?= /usr/local/cuda
HAPICC = $(HAPITOOLKIT_HOME)/bin/nvcc
HAPICC_FLAGS = -c -std=c++14
HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
HAPICC_LIB = -L$(HAPITOOLKIT_HOME)/lib64

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
	$(HAPICC) -o $@ $(HAPICC_FLAGS) $(HAPICC_INC) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun

test: all
	$(call run, ./$(TARGET) +p2)
