OPTS = -O3 -DHAPI_CUDA_CALLBACK

# This charm links against LCI, and liblci.so has a transitive dependency on
# liblct.so that ld will not resolve on its own.
export LD_LIBRARY_PATH := $(HOME)/lci_install/lib:$(HOME)/lci_install_cuda/lib:$(LD_LIBRARY_PATH)

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

NVCC = nvcc
NVCC_FLAGS = -c -std=c++11 $(OPTS)
LD_LIBS = -module EveryLB

TARGET = leanmd
all: $(TARGET)

OBJS = Main.o Cell.o Compute.o leanmdCUDA.o

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
