CHARMDIR=../..
-include $(CHARMDIR)/include/conv-mach-opt.mak

CUDA_PATH := $(CUDA_DIR)
CHARMC    := $(CHARMDIR)/bin/charmc
FLAGS     := $(OPTSATBUILDTIME)
INC       := -I$(CUDA_PATH)/include -I..

all: libs
	cp libhybridapi.a $(CHARMDIR)/lib

libs: libhybridapi.a

install: libhybridapi.a
	cp libhybridapi.a $(CHARMDIR)/lib

libhybridapi.a: hybridapi.o buddy_allocator.o ck.o
	-rm -f $@
	ar q $@ hybridapi.o buddy_allocator.o ck.o

hybridapi.o: hapi_impl.cpp hapi_impl.h gpumanager.h devicemanager.h buddy_allocator.h hapi.h hapi_nvtx.h
	$(CHARMC) $(FLAGS) $(INC) -o $@ -c $<

buddy_allocator.o : buddy_allocator.cpp buddy_allocator.h
	$(CHARMC) $(FLAGS) $(INC) -o $@ -c $<

clean:
	rm -f *.o *.a
