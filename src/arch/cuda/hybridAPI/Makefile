CHARMDIR=../..
-include $(CHARMDIR)/include/conv-mach-opt.mak

CUDA_PATH := $(CUDA_DIR)
CHARMC    := $(CHARMDIR)/bin/charmc
FLAGS     := $(OPTSATBUILDTIME) -DHAPI_TRACE
INC       := -I$(CUDA_PATH)/include -I$(CUDA_PATH)/extras/CUPTI/include -I..

all: libs
	cp libcudahybridapi.a $(CHARMDIR)/lib

libs: libcudahybridapi.a

install: libcudahybridapi.a
	cp libcudahybridapi.a $(CHARMDIR)/lib

libcudahybridapi.a: hybridapi.o hapi_cupti.o
	-rm -f $@
	ar q $@ $^

hybridapi.o: hapi_impl.cpp hapi_impl.h hapi.h hapi_nvtx.h
	$(CHARMC) $(FLAGS) $(INC) -o $@ $<

hapi_cupti.o: hapi_cupti.cpp hapi_cupti.h
	$(CHARMC) $(FLAGS) $(INC) -o $@ $<

clean:
	rm -f *.o *.a
