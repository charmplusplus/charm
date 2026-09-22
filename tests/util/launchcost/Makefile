# Standalone CUDA microbenchmark: no Charm++ dependency. Host cost of a driver
# call from 1 vs T threads, CUPTI tracing off/on (see launchcost.cu).
NVCC ?= nvcc
ARCH ?= sm_86
CUPTI ?= /opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/extras/CUPTI
all: launchcost
launchcost: launchcost.cu
	$(NVCC) -O2 -arch=$(ARCH) -I$(CUPTI)/include -o $@ launchcost.cu -L$(CUPTI)/lib64 -Xlinker -rpath -Xlinker $(CUPTI)/lib64 -lcupti -lcuda -lpthread
clean:
	rm -f launchcost
