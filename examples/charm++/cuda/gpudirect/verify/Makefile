OPTS = -O0 -g

# GPU backend to build for: "cuda" (default) or "hip".
GPU ?= cuda

ifeq ($(GPU),hip)
  CHARM_BUILD ?= reconverse-linux-x86_64-amd
  GPU_FLAGS = -D__HIP_PLATFORM_AMD__=ON
  HAPITOOLKIT_HOME ?= $(if $(ROCM_PATH),$(ROCM_PATH),/opt/rocm)
  HAPICC = hipcc
  HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
  HAPICC_LIB = -L$(HAPITOOLKIT_HOME)/lib
  HAPI_ARCH_FLAGS =
  LD_LIBS =
else ifeq ($(GPU),cuda)
  CHARM_BUILD ?= reconverse-linux-x86_64-cuda
  GPU_FLAGS =
  HAPITOOLKIT_HOME ?= $(if $(CUDATOOLKIT_HOME),$(CUDATOOLKIT_HOME),$(if $(CUDA_HOME),$(CUDA_HOME),/usr/local/cuda))
  HAPICC = $(HAPITOOLKIT_HOME)/bin/nvcc
  HAPICC_INC = -I$(HAPITOOLKIT_HOME)/include
  HAPICC_LIB = -L$(HAPITOOLKIT_HOME)/lib64
  # sm_86 is the A40; override CUDA_ARCH for other devices (A100 is sm_80).
  CUDA_ARCH ?= sm_86
  # CUDA <= 11.8 rejects host compilers newer than GCC 11, so allow the host
  # compiler to be pinned independently of the one Charm++ itself was built with.
  CUDA_HOST_CC ?=
  HAPI_ARCH_FLAGS = -arch=$(CUDA_ARCH) $(if $(CUDA_HOST_CC),-ccbin $(CUDA_HOST_CC),)
  LD_LIBS = -L$(HAPITOOLKIT_HOME)/lib64 -lcudart
else
  $(error GPU must be one of "cuda" or "hip", got "$(GPU)")
endif

# Override on the command line to build against a different Charm++ build,
# e.g. make CHARM_DIR=/path/to/reconverse-linux-x86_64-cuda
CHARM_DIR ?= ../../../../../$(CHARM_BUILD)
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

HAPI_FLAGS = -O3 -g -c -std=c++11 -use_fast_math

TARGET = verify
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) $(GPU_FLAGS) -language charm++ -module CommonLBs -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $(GPU_FLAGS) $<

$(TARGET).def.h: $(TARGET).ci

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h
	$(CHARMC) $(GPU_FLAGS) -c $<

$(TARGET)CUDA.o: $(TARGET).cu
	$(HAPICC) $(GPU_FLAGS) -o $@ $(HAPI_FLAGS) $(HAPI_ARCH_FLAGS) $(HAPICC_INC) $(CHARM_INC) $(HAPICC_LIB) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
