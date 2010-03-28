MTL       = ../mtl
EXEC      = par_minisat

#CHARMDIR = /Users/yanhuasun/PPL/charm/charm-6.1.2/multicore-darwin-x86_64
OPTS = -g
CXX ?= charmc $(OPTS)  
default: all
all: sat


CXX       ?= g++
CFLAGS    ?= -I$(MTL) -Wall -ffloat-store
LFLAGS    ?= -Wall -lz -language charm++

COPTIMIZE ?= -O3

## Compile options
%.o:			CFLAGS +=$(COPTIMIZE) -ggdb -D DEBUG


CSRCS     ?= $(wildcard *.C)
CHDRS     ?= $(wildcard *.h)
COBJS     ?= $(addsuffix .o, $(basename $(CSRCS)))


$(EXEC):        $(COBJS)

## Build rule
%.o:	%.C
	@echo Compiling: "$@ ( $< )"
	@$(CXX) $(CFLAGS) -c -o $@ $<

## Linking rules (standard/profile/debug/release)
$(EXEC):
	@echo Linking: "$@ ( $^ )"
	 @$(CXX) $^ $(LFLAGS) -o $@

main.o : main.C main.decl.h main.def.h 

par_Solver.o : par_Solver.C par_SolverTypes.h par_Solver.h Solver.C Solver.h

main.def.h main.decl.h : main.ci
	$(CHARMC) main.ci

Solver.o: Solver.C Solver.h ../mtl/Vec.h ../mtl/Heap.h ../mtl/Vec.h \
  ../mtl/Alg.h SolverTypes.h ../mtl/Sort.h ../mtl/Vec.h

## Clean rule
clean:
	@rm -f $(EXEC) $(EXEC)_profile $(EXEC)_debug $(EXEC)_release $(EXEC)_static \
	  $(COBJS) $(PCOBJS) $(DCOBJS) $(RCOBJS) *.core depend.mak lib$(LIB).a lib$(LIB)d.a


