OPTS = -O3 -DHAPI_CUDA_CALLBACK

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

NVCC = nvcc
NVCC_FLAGS = -c -std=c++17 -use_fast_math $(OPTS)
LD_LIBS = -module EveryLB -module DiffusionLB

TARGET = sph3d
all: $(TARGET)

OBJS = $(TARGET).o $(TARGET)CUDA.o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci $(TARGET).h
	$(CHARMC) $<

$(TARGET).def.h: $(TARGET).ci $(TARGET).h

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h $(TARGET).h
	$(CHARMC) -c $<

$(TARGET)CUDA.o: $(TARGET).cu $(TARGET).h
	$(NVCC) -o $@ $(NVCC_FLAGS) $(CHARM_INC) $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun

# Kokkos variant (sph3d_kokkos.cpp, ../hapi_kokkos.h): the same host object, the kernels as Kokkos functors; CUB's
# device scan/select stay. Same device flags as the CUDA variant plus C++20 and extended lambdas (Kokkos develop).
-include $(CHARM_DIR)/include/conv-mach-opt.mak
# Kokkos 5 (the develop branch) needs C++20; 4.7 builds with it too.
KOKKOS_STD ?= c++20
# Kokkos sources are plain .cpp; the vendor's compiler is told the language and gets its own flags: nvcc -x cu with
# extended lambdas on CUDA, hipcc -x hip on HIP (lambdas need no flag there). The Kokkos install follows the vendor.
ifeq ($(BUILD_HIP),1)
KOKKOS_DIR ?= /u/bhosale/kokkos-develop/install-hip
KOKKOS_CXX = hipcc -x hip $(patsubst %,--offload-arch=%,$(subst ;, ,$(HIP_ARCHS)))
KOKKOS_DEV_FLAGS = -ffast-math
KOKKOS_DEV_LIBS =
else
KOKKOS_DIR ?= /u/bhosale/kokkos-develop/install-sm86
KOKKOS_CXX = nvcc -x cu
KOKKOS_DEV_FLAGS = -use_fast_math -extended-lambda -Wext-lambda-captures-this
KOKKOS_DEV_LIBS = -lcuda
endif
KOKKOS_NVCC_FLAGS = -c -std=$(KOKKOS_STD) $(KOKKOS_DEV_FLAGS) $(OPTS) -I$(KOKKOS_DIR)/include -I..
KOKKOS_LIBS = -L$(KOKKOS_DIR)/lib64 -lkokkoscore -ldl $(KOKKOS_DEV_LIBS)

$(TARGET).kokkos: $(TARGET).o $(TARGET)KOKKOS.o
	$(CHARMC) -language charm++ -o $@ $^ $(LD_LIBS) $(KOKKOS_LIBS)

$(TARGET)KOKKOS.o: $(TARGET)_kokkos.cpp $(TARGET).h ../hapi_kokkos.h
	$(KOKKOS_CXX) -o $@ $(KOKKOS_NVCC_FLAGS) $(CHARM_INC) $<
