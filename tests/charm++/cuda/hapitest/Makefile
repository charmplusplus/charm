# tests/charm++/cuda/hapitest -- stage 9.1 acceptance test.
#
# Builds against either GPU backend from one source; pick with GPU=cuda (the
# default) or GPU=hip. Only hapitest.cu goes through nvcc/hipcc -- hapitest.C is
# compiled by charmc on purpose, so that hapi_portable.h is exercised the way
# user code actually sees it.
#
# CHARM_DIR defaults to ../../../.. , which is the Charm++ build root when this
# directory is reached through the build tree (<build>/tests/charm++/cuda/
# hapitest) -- the same convention every other test and example Makefile uses.
# Building straight out of the source tree needs it spelled out:
#
#   make GPU=hip  CHARM_DIR=../../../../reconverse-linux-x86_64-amd
#   make GPU=cuda CHARM_DIR=../../../../netlrts-linux-x86_64-cuda CUDA_ARCH=sm_80

OPTS = -O2 -g

GPU ?= cuda

ifeq ($(GPU),hip)
  GPU_FLAGS = -D__HIP_PLATFORM_AMD__=ON
  HAPITOOLKIT_HOME ?= $(if $(ROCM_PATH),$(ROCM_PATH),/opt/rocm)
  HAPICC = $(HAPITOOLKIT_HOME)/bin/hipcc
  HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
  # gfx90a is the MI250X (Frontier); override HIP_ARCH elsewhere.
  HIP_ARCH ?= gfx90a
  # -ffp-contract=off: the result check in hapitest.C is an exact double
  # comparison, so the kernel must not contract its multiply-add into an FMA
  # with different rounding from the host expression.
  HAPI_ARCH_FLAGS = --offload-arch=$(HIP_ARCH) -ffp-contract=off
  LD_LIBS =
else ifeq ($(GPU),cuda)
  GPU_FLAGS =
  HAPITOOLKIT_HOME ?= $(if $(CUDATOOLKIT_HOME),$(CUDATOOLKIT_HOME),$(if $(CUDA_HOME),$(CUDA_HOME),/usr/local/cuda))
  HAPICC = $(HAPITOOLKIT_HOME)/bin/nvcc
  HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
  # sm_80 is the A100; override CUDA_ARCH for other devices.
  CUDA_ARCH ?= sm_80
  # -fmad=false: see the HIP branch -- keep the kernel from contracting its
  # multiply-add, which the exact result comparison would notice.
  HAPI_ARCH_FLAGS = -arch=$(CUDA_ARCH) -fmad=false
  LD_LIBS =
else
  $(error GPU must be one of "cuda" or "hip", got "$(GPU)")
endif

CHARM_DIR ?= ../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

HAPI_FLAGS = -O2 -g -c -std=c++14

TARGET = hapitest
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) $(GPU_FLAGS) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h $(TARGET).def.h: $(TARGET).ci
	$(CHARMC) $(GPU_FLAGS) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h
	$(CHARMC) $(GPU_FLAGS) -c $<

$(TARGET)CUDA.o: $(TARGET).cu
	$(HAPICC) $(GPU_FLAGS) -o $@ $(HAPI_FLAGS) $(HAPI_ARCH_FLAGS) $(HAPICC_INC) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun

test: all
	./$(TARGET) +pe 4
