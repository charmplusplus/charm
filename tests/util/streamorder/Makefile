# Standalone CUDA microbenchmark: no Charm++ dependency. -lcuda is for the
# driver-API memset the pinned-flag completion arm uses (cuMemsetD32Async).
NVCC ?= nvcc
ARCH ?= sm_86
all: streamorder
streamorder: streamorder.cu
	$(NVCC) -O2 -arch=$(ARCH) -o $@ streamorder.cu -lcuda
clean:
	rm -f streamorder
# In-process setenv placement check for the runtime's CUDA_DEVICE_MAX_CONNECTIONS preset.
connpreset: connpreset.cu
	$(NVCC) -O2 -arch=$(ARCH) -o $@ connpreset.cu -lcuda
