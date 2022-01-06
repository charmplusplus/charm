-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
#CHARMC ?=
USEROPTS= -O3 -g
#-lpapi
CHARMDIR=../../../..
CHARMINC=$(CHARMDIR)/include
OPTS=-I$(CHARMINC) $(USEROPTS)
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS)
CHARMLIB=$(CHARMDIR)/lib
CXX = ${CHARMC}
CC = ${CXX}
DOTPROD = dotProd
DOTPRODOMP = dotProd_withomp

TARGETS := $(DOTPROD)
ifeq (1,$(CMK_HAS_OPENMP))
  TARGETS += $(DOTPRODOMP)
endif

all: $(TARGETS)

dotProd: dotProd.o
	$(CHARMC) -language charm++ -o dotProd dotProd.o -module CkLoop
#-DPAPI_PROFILING -DCKL_HYBRID

test: $(addprefix test-,$(TARGETS))

testp: $(addprefix testp-,$(TARGETS))

test-dotProd: dotProd
	$(call run, +p4 ./$< 800 2 0.7 5 )
	$(call run, +p4 ./$< 800 4 0.7 5 )
ifeq ($(CMK_SMP),1)
ifneq ($(CMK_MULTICORE),1)
	$(call run, +p4 ./$< 800 2 0.7 5 ++ppn 2 )
	$(call run, +p4 ./$< 800 4 0.7 5 ++ppn 2 )
endif
endif

test-dotProd_withomp: dotProd_withomp
	$(call run, +p4 ./$< 800 2 0.7 5 )
	$(call run, +p4 ./$< 800 4 0.7 5 )
ifeq ($(CMK_SMP),1)
ifneq ($(CMK_MULTICORE),1)
	$(call run, +p4 ./$< 800 2 0.7 5 ++ppn 2 )
	$(call run, +p4 ./$< 800 4 0.7 5 ++ppn 2 )
endif
endif

testp-dotProd: dotProd
	$(call run, +p$(P) ./$< 800 $$(( $(P) / 2 )) 0.7 5 )
	$(call run, +p$(P) ./$< 800 $(P) 0.7 5)

testp-dotProd_withomp: dotProd_withomp
	$(call run, +p$(P) ./$< 800 $$(( $(P) / 2 )) 0.7 5 )
	$(call run, +p$(P) ./$< 800 $(P) 0.7 5 )

dotProd_withomp: dotProd_withomp.o
	$(CHARMC) -language charm++ -o dotProd_withomp dotProd_withomp.o -module CkLoop -fopenmp

%.decl.h: %.ci
	$(CXX) $<

dotProd.decl.h: dotProd.ci
	$(CHARMC) dotProd.ci

dotProd.o: dotProd.C dotProd.decl.h
	$(CHARMC) -c dotProd.C

dotProd_withomp.o: dotProd.C dotProd.decl.h
	$(CHARMC) -c dotProd.C -fopenmp -o dotProd_withomp.o

${DOTPROD}.o: ${DOTPROD}.decl.h

${DOTPRODOMP}.o: ${DOTPROD}.decl.h

clean:
	rm -rf $(wildcard *.o *decl.h *def.h charmrun $(DOTPROD) $(DOTPRODOMP))
