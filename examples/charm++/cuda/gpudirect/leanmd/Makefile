OPTS = -O3 -DHAPI_CUDA_CALLBACK

# This charm links against LCI, and liblci.so has a transitive dependency on
# liblct.so that ld will not resolve on its own.
export LD_LIBRARY_PATH := $(HOME)/lci_install/lib64:$(HOME)/lci_install/lib:$(HOME)/lci_install_cuda/lib64:$(HOME)/lci_install_cuda/lib:$(LD_LIBRARY_PATH)

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

# The device compiler follows the build the runtime was made with (conv-mach-opt.mak):
# nvcc for a CUDA build, hipcc as HIP for a HIP build (kernels in CUDA syntax compile
# as HIP; the host side uses the hapi names of hapi_portable.h).
-include $(CHARM_DIR)/include/conv-mach-opt.mak
ifeq ($(BUILD_HIP),1)
NVCC = hipcc
NVCC_FLAGS = -x hip -c -std=c++17 $(OPTS) $(patsubst %,--offload-arch=%,$(subst ;, ,$(HIP_ARCHS)))
else
NVCC = nvcc
NVCC_FLAGS = -c -std=c++11 $(OPTS)
endif
LD_LIBS = -module EveryLB -module DiffusionLB

TARGET = leanmd
all: $(TARGET)

OBJS = Main.o Cell.o Compute.o leanmdCUDA.o

Main.o Cell.o Compute.o: JoinTrace.h

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

leanmd.decl.h leanmd.def.h: leanmd.ci
	$(CHARMC) leanmd.ci

Main.o: Main.cc Main.h leanmd.decl.h leanmd.def.h defs.h Cell.h Compute.h
	$(CHARMC) -c $<

Cell.o: Cell.cc Cell.h leanmd.decl.h defs.h leanmd_cuda.h
	$(CHARMC) -c $<

Compute.o: Compute.cc Compute.h Cell.h leanmd.decl.h defs.h leanmd_cuda.h
	$(CHARMC) -c $<

# The kernels see the generated headers too, because defs.h pulls in leanmd.decl.h
# for the proxy declarations.
leanmdCUDA.o: leanmd.cu leanmd_cuda.h defs.h leanmd.decl.h
	$(NVCC) -o $@ $(NVCC_FLAGS) $(CHARM_INC) $<

test: $(TARGET)
	./charmrun +p4 ./$(TARGET) 3 3 3 100 20 20 +balancer GreedyRefineLB ++local

clean:
	rm -f *.decl.h *.def.h *.o $(TARGET) charmrun

# Kokkos variant (leanmd_kokkos.cpp, ../hapi_kokkos.h): the same host objects, the kernels as Kokkos functors. Same device
# flags as the CUDA variant (no -arch: sm_75 code with PTX, as leanmdCUDA.o has) so an A/B differs only in the kernel file.
# Kokkos develop (5.2.99 + the perf/launch-path commits of /u/bhosale/kokkos-develop) built for sm_86 at KOKKOS_DIR; the user's 4.7.4 at /u/bhosale/kokkos/install-sm86 builds too with KOKKOS_STD=c++17.
# Kokkos 5 (the develop branch) needs C++20; 4.7 builds with it too.
KOKKOS_STD ?= c++20
# Kokkos sources are plain .cpp; the vendor's compiler is told the language and gets its own flags: nvcc -x cu with
# extended lambdas on CUDA, hipcc -x hip on HIP (lambdas need no flag there). The Kokkos install follows the vendor.
ifeq ($(BUILD_HIP),1)
KOKKOS_DIR ?= /u/bhosale/kokkos-develop/install-hip
KOKKOS_CXX = hipcc -x hip $(patsubst %,--offload-arch=%,$(subst ;, ,$(HIP_ARCHS)))
KOKKOS_DEV_FLAGS = 
KOKKOS_DEV_LIBS =
else
KOKKOS_DIR ?= /u/bhosale/kokkos-develop/install-sm86
KOKKOS_CXX = nvcc -x cu
KOKKOS_DEV_FLAGS = -extended-lambda -Wext-lambda-captures-this
KOKKOS_DEV_LIBS = -lcuda
endif
KOKKOS_NVCC_FLAGS = -c -std=$(KOKKOS_STD) $(KOKKOS_DEV_FLAGS) $(OPTS) -I$(KOKKOS_DIR)/include -I..
KOKKOS_LIBS = -L$(KOKKOS_DIR)/lib64 -lkokkoscore -ldl $(KOKKOS_DEV_LIBS)

$(TARGET).kokkos: Main.o Cell.o Compute.o leanmdKOKKOS.o
	$(CHARMC) -language charm++ -o $@ $^ $(LD_LIBS) $(KOKKOS_LIBS)

leanmdKOKKOS.o: leanmd_kokkos.cpp leanmd_cuda.h defs.h leanmd.decl.h ../hapi_kokkos.h
	$(KOKKOS_CXX) -o $@ $(KOKKOS_NVCC_FLAGS) $(CHARM_INC) $<
