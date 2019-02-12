-include ../../../../common.mk
include  ../../../../../include/conv-mach-opt.mak
CHARMC = ../../../../../bin/charmc $(OPTS)

OMP_OPT_GCC=-fopenmp
OMP_OPT_CLANG=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
OMP_OPT_XL=-qsmp=omp -qoffload

ifeq ($(CMK_COMPILER),gcc)
	OMP_OPT=$(OMP_OPT_GCC)
else ifeq ($(CMK_COMPILER),clang)
	OMP_OPT=$(OMP_OPT_CLANG)
else ifeq ($(CMK_COMPILER),xlc)
	OMP_OPT=$(OMP_OPT_XL)
else
	OMP_OPT=-fopenmp
endif


OBJS = hello.o

all: hello

hello: $(OBJS)
	$(CHARMC) -language charm++ $(OMP_OPT) -o hello $(OBJS)

hello.decl.h: hello.ci
	$(CHARMC) hello.ci

clean:
	rm -f *.decl.h *.def.h *.o hello charmrun

hello.o: hello.C hello.decl.h
	$(CHARMC) $(OMP_OPT) -c hello.C

test: all
	$(call run, ./hello)
