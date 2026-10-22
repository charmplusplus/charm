OPTS = -O3

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

# The device compiler follows the build the runtime was made with (conv-mach-opt.mak):
# nvcc for a CUDA build, hipcc as HIP for a HIP build (kernels in CUDA syntax compile
# as HIP; the host side uses the hapi names of hapi_portable.h).
-include $(CHARM_DIR)/include/conv-mach-opt.mak
ifeq ($(BUILD_HIP),1)
NVCC = hipcc
NVCC_FLAGS = -x hip -c -std=c++17 -ffast-math $(OPTS) $(patsubst %,--offload-arch=%,$(subst ;, ,$(HIP_ARCHS)))
else
NVCC = nvcc
NVCC_FLAGS = -c -std=c++11 -use_fast_math $(OPTS)
endif
LD_LIBS =

TARGET = jacobi3d
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

# Kokkos variant (jacobi3d_kokkos.cpp, hapi_kokkos.h): the same host objects, the kernels as Kokkos functors. Same device
# flags as the CUDA variant (no -arch: sm_75 code with PTX, as leanmdCUDA.o/jacobi3dCUDA.o have) so an A/B differs only
# in the kernel file. Kokkos develop (5.2.99 + the perf/launch-path commits of /u/bhosale/kokkos-develop) built for sm_86 at KOKKOS_DIR; the user's 4.7.4 at /u/bhosale/kokkos/install-sm86 builds too with KOKKOS_STD=c++17.
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
