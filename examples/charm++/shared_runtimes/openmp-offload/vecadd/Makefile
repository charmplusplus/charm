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


OBJS = vecadd.o

all: vecadd

vecadd: $(OBJS)
	$(CHARMC) -language charm++ $(OMP_OPT) -o vecadd $(OBJS)

vecadd.decl.h: vecadd.ci
	$(CHARMC) vecadd.ci

clean:
	rm -f *.decl.h *.def.h *.o vecadd charmrun

vecadd.o: vecadd.C vecadd.decl.h
	$(CHARMC) $(OMP_OPT) -c vecadd.C

test: all
	$(call run, ./vecadd +p2)
