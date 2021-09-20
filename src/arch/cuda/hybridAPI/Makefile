CHARMDIR=../..
-include $(CHARMDIR)/include/conv-mach-opt.mak

CUDA_PATH := $(CUDA_DIR)
CHARMC    := $(CHARMDIR)/bin/charmc
FLAGS     := $(OPTSATBUILDTIME)
INC       := -I$(CUDA_PATH)/include -I..

all: libs
	cp libcudahybridapi.a $(CHARMDIR)/lib

libs: libcudahybridapi.a

install: libcudahybridapi.a
	cp libcudahybridapi.a $(CHARMDIR)/lib

libcudahybridapi.a: hybridapi.o buddy_allocator.o
	-rm -f $@
	ar q $@ hybridapi.o buddy_allocator.o

hybridapi.o: hapi_impl.cpp hapi_impl.h gpumanager.h devicemanager.h buddy_allocator.h hapi.h hapi_nvtx.h
	$(CHARMC) $(FLAGS) $(INC) -o $@ -c $<

buddy_allocator.o : buddy_allocator.cpp buddy_allocator.h
	$(CHARMC) $(FLAGS) $(INC) -o $@ -c $<

clean:
	rm -f *.o *.a
