# Host-only: the range heap has no CUDA or Charm++ dependency.
CXX ?= g++
all: rangeheap_test
rangeheap_test: rangeheap_test.C ../../../src/arch/cuda/hybridAPI/hapi_range_heap.h
	$(CXX) -std=c++14 -O2 -Wall -o $@ rangeheap_test.C
test: rangeheap_test
	./rangeheap_test
clean:
	rm -f rangeheap_test
