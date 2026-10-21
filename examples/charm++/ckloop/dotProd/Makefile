-include ../../../common.mk
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

default: ${DOTPRODOMP}

dotProd: dotProd.o
	$(CHARMC) -language charm++ -o dotProd dotProd.o -module CkLoop
#-DPAPI_PROFILING -DCKL_HYBRID

test1: dotProd
	$(call run, ./dotProd +p4 800 2 0.7 5)

test2: dotProd
	$(call run, ./dotProd +p4 800 2 0.7 5)

test3: dotProd
	$(call run, ./dotProd +p4 800 2 0.7 5)

test4: dotProd
	$(call run, ./dotProd +p4 800 4 0.7 5)

test5: dotProd
	$(call run, ./dotProd +p6 800 2 0.7 5)

dotProd_withomp: dotProd_withomp.o
	$(CHARMC) -language charm++ -o dotProd_withomp dotProd_withomp.o -fopenmp

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

module: $(CHARMLIB)/libmoduleCkLoop.a

$(CHARMLIB)/libmoduleCkLoop.a: CkLoop.o
	$(CHARMC)  -o $(CHARMLIB)/libmoduleCkLoop.a CkLoop.o

CkLoop.decl.h: CkLoop.ci
	$(CHARMC) CkLoop.ci

CkLoop.o: CkLoop.C CkLoop.decl.h
	$(CHARMC) -c CkLoop.C

clean:
	rm -rf $(wildcard *.o *decl.h *def.h charmrun $(DOTPROD) $(DOTPRODOMP))
