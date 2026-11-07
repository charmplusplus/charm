# rma_bw: LCI-level put-against-get bandwidth between two hosts (see rma_bw.cpp).
# Not a Charm++ program: it links LCI and the vendor runtime only.
#   make                      CUDA, LCI from $(LCI_ROOT) (an install with include/ and lib64/ or lib/)
#   make BUILD_HIP=1          HIP (hipcc), the HIP LCI install in $(LCI_ROOT)
# Run with one rank per host, inside a two-node allocation, for example on Delta:
#   srun -N 2 -n 2 --ntasks-per-node=1 ./rma_bw put dd 134217728 10 both
LCI_ROOT ?= /u/bhosale/lci_install
LCI_LIB  ?= $(firstword $(wildcard $(LCI_ROOT)/lib64 $(LCI_ROOT)/lib))
ifeq ($(BUILD_HIP),1)
CXX      ?= hipcc
CXXFLAGS += -DRMA_BW_HIP
DEVLIBS   = -lamdhip64
else
CUDA_HOME ?= $(shell dirname $$(dirname $$(command -v nvcc)))
CXX      ?= g++
CXXFLAGS += -I$(CUDA_HOME)/include
DEVLIBS   = -L$(CUDA_HOME)/lib64 -lcudart
endif
CXXFLAGS += -O2 -std=c++17 -I$(LCI_ROOT)/include

rma_bw: rma_bw.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(LCI_LIB) -Wl,-rpath,$(LCI_LIB) -llci -llct $(DEVLIBS)

clean:
	rm -f rma_bw
