# Host-only: the range heap has no CUDA or Charm++ dependency, so this builds
# with the host compiler, not charmc, and runs as a plain program whatever the
# test launcher settings. The header is found through the source file's real
# path, which also holds in a build tree whose tests are symlinks.
CXX ?= g++
HEAPDIR := $(dir $(realpath rangeheap_test.C))../../../src/arch/cuda/hybridAPI

all: rangeheap_test

rangeheap_test: rangeheap_test.C $(HEAPDIR)/hapi_range_heap.h
	$(CXX) -std=c++14 -O2 -Wall -I$(HEAPDIR) -o $@ rangeheap_test.C

test: rangeheap_test
	./rangeheap_test

testp: test

clean:
	rm -f rangeheap_test
