-include ../../../../common.mk

OPTS = -O3

# Override to build against a different Charm++ build, e.g.
#   make CHARM_DIR=../../../../../reconverse-linux-x86_64-amd
CHARM_DIR ?= ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include
LD_LIBS =

# GPU backend to build for: "cuda" or "hip". Defaults to whatever the Charm++
# build in CHARM_DIR was configured with -- cmake appends that backend's
# conv-mach header to conv-mach-opt.h -- so a recursive build in either tree
# needs no flags. Override with GPU=cuda or GPU=hip.
GPU ?= $(if $(shell grep -l conv-mach-hip.h $(CHARM_DIR)/include/conv-mach-opt.h 2>/dev/null),hip,cuda)

ifeq ($(GPU),hip)
  # charmc is a plain host compiler here, so user code needs the platform macro
  # spelled out; hipcc sets it for itself.
  GPU_FLAGS = -D__HIP_PLATFORM_AMD__=ON
  HAPITOOLKIT_HOME ?= $(if $(ROCM_PATH),$(ROCM_PATH),/opt/rocm)
  HAPICC = $(HAPITOOLKIT_HOME)/bin/hipcc
  HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
  # gfx90a is the MI250X (Frontier); override HIP_ARCH for other devices.
  HIP_ARCH ?= gfx90a
  HAPI_ARCH_FLAGS = --offload-arch=$(HIP_ARCH)
  HAPI_FAST_MATH = -ffast-math
else ifeq ($(GPU),cuda)
  GPU_FLAGS =
  HAPITOOLKIT_HOME ?= $(if $(CUDATOOLKIT_HOME),$(CUDATOOLKIT_HOME),$(if $(CUDA_HOME),$(CUDA_HOME),/usr/local/cuda))
  HAPICC = $(HAPITOOLKIT_HOME)/bin/nvcc
  HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
  # GPU architecture the kernels are compiled for. Without this nvcc emits only
  # its default target (sm_75 as of CUDA 13), and on a newer device the kernel
  # can fail to launch with nothing but an unchecked error to show for it.
  # sm_80 is the A100; override CUDA_ARCH for other devices (sm_86 A40, sm_90 H100).
  CUDA_ARCH ?= sm_80
  HAPI_ARCH_FLAGS = -arch=$(CUDA_ARCH)
  HAPI_FAST_MATH = -use_fast_math
else
  $(error GPU must be one of "cuda" or "hip", got "$(GPU)")
endif

HAPI_FLAGS = -c -std=c++11 $(HAPI_FAST_MATH) $(OPTS)

TARGET = jacobi2d
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) $(GPU_FLAGS) -language charm++ -module CommonLBs -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h $(TARGET).def.h: $(TARGET).ci $(TARGET).h
	$(CHARMC) $(GPU_FLAGS) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h $(TARGET).h
	$(CHARMC) $(GPU_FLAGS) -c $<

$(TARGET)CUDA.o: $(TARGET).cu $(TARGET).h
	$(HAPICC) $(GPU_FLAGS) -o $@ $(HAPI_FLAGS) $(HAPI_ARCH_FLAGS) $(HAPICC_INC) $(CHARM_INC) $<

test: all
	$(call run, ./$(TARGET) +p4)

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
