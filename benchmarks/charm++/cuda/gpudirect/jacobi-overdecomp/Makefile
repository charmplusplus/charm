# benchmarks/charm++/cuda/gpudirect/jacobi-overdecomp -- jacobi_od.
#
# A portable (CUDA and HIP) copy of examples/charm++/cuda/gpudirect/jacobi2d,
# kept separate so the example stays untouched. One source set builds against
# either GPU backend; pick with GPU=cuda (the default) or GPU=hip. Only
# jacobi_od.cu goes through nvcc/hipcc -- jacobi_od.C is compiled by charmc, so
# the headers are exercised the way user code actually sees them.
#
# CHARM_DIR defaults to ../../../../.. , which is the Charm++ build root when
# this directory is reached through the build tree (<build>/benchmarks/charm++/
# cuda/gpudirect/jacobi-overdecomp) -- the convention the other benchmark and
# example Makefiles use. Building straight out of the source tree needs it
# spelled out:
#
#   make GPU=cuda CHARM_DIR=../../../../../reconverse-linux-x86_64-cuda CUDA_ARCH=sm_80
#   make GPU=hip  CHARM_DIR=../../../../../reconverse-linux-x86_64-amd  GPU_ARCH=gfx90a
#
# Optional: -DTEST_CORRECTNESS switches DataType to int (see jacobi_od.h);
# -DHAPI_NVTX_PROFILE turns on the NVTX ranges (CUDA only).

# Same optional include the sibling gpudirect benchmarks use (it exists in the
# build tree, not in the source tree -- hence -include).
-include ../../../../common.mk

OPTS = -O3

GPU ?= cuda

# GPU_ARCH is the portable spelling of the device target; CUDA_ARCH is still
# accepted for the CUDA branch so the flag names match the original example.
ifeq ($(GPU),hip)
  GPU_FLAGS = -D__HIP_PLATFORM_AMD__=ON
  HAPITOOLKIT_HOME ?= $(if $(ROCM_PATH),$(ROCM_PATH),/opt/rocm)
  HAPICC = $(HAPITOOLKIT_HOME)/bin/hipcc
  HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
  # gfx90a is the MI250X (Frontier); override GPU_ARCH elsewhere.
  GPU_ARCH ?= gfx90a
  # No -use_fast_math here: that is an nvcc spelling. hipcc's equivalent is
  # -ffast-math, deliberately left off so the HIP and CUDA builds do the same
  # arithmetic.
  HAPI_ARCH_FLAGS = --offload-arch=$(GPU_ARCH)
  LD_LIBS =
else ifeq ($(GPU),cuda)
  GPU_FLAGS =
  HAPITOOLKIT_HOME ?= $(if $(CUDATOOLKIT_HOME),$(CUDATOOLKIT_HOME),$(if $(CUDA_HOME),$(CUDA_HOME),/usr/local/cuda))
  HAPICC = $(HAPITOOLKIT_HOME)/bin/nvcc
  HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
  # Without an explicit -arch nvcc emits only its default target and on a newer
  # device the kernel can fail to launch -- which the CHECK_LAUNCH in
  # jacobi_od.cu now reports instead of silently producing wrong numbers.
  # sm_80 is the A100; override CUDA_ARCH (sm_86 A40, sm_90 H100). GPU_ARCH is
  # accepted as an alias.
  CUDA_ARCH ?= $(if $(GPU_ARCH),$(GPU_ARCH),sm_80)
  HAPI_ARCH_FLAGS = -arch=$(CUDA_ARCH) -use_fast_math
  LD_LIBS =
else
  $(error GPU must be one of "cuda" or "hip", got "$(GPU)")
endif

CHARM_DIR ?= ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

HAPI_FLAGS = -c -std=c++14 $(OPTS)

TARGET = jacobi_od
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)GPU.o

$(TARGET): $(OBJS)
	$(CHARMC) $(GPU_FLAGS) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

# charmc emits both headers from one run; the second target carries no recipe
# so make does not invoke the translator twice (the idiom the example uses).
$(TARGET).decl.h: $(TARGET).ci $(TARGET).h
	$(CHARMC) $(GPU_FLAGS) $<

$(TARGET).def.h: $(TARGET).decl.h

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h $(TARGET).h
	$(CHARMC) $(GPU_FLAGS) -c $<

$(TARGET)GPU.o: $(TARGET).cu $(TARGET).h
	$(HAPICC) $(GPU_FLAGS) -o $@ $(HAPI_FLAGS) $(HAPI_ARCH_FLAGS) $(HAPICC_INC) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun

# Smallest meaningful over-decomposed run: 4x4 chares on 2 PEs in one process.
test: all
	./$(TARGET) -W 512 -H 512 -w 128 -h 128 -i 10 -u 2 +pe 2
