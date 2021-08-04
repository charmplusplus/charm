-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
CHARMDIR=../../../..
CHARMINC=$(CHARMDIR)/include
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS) -fopenmp
CHARMLIB=$(CHARMDIR)/lib

TARGETS :=
ifeq (1,$(CMK_HAS_OPENMP))
  TARGETS += hello
endif

all: $(TARGETS)

hello: hello.o
	$(CHARMC) -language charm++ -o hello hello.o  -module CkLoop

hello.prj: hello.o
	$(CHARMC) -language charm++ -o hello.prj hello.o  -tracemode projections -module CkLoop

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

hello.o: hello.C hello.decl.h 
	$(CHARMC) -c hello.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello hello.prj charmrun *.log *.sum *.sts

test: $(addprefix test-,$(TARGETS))

testp: $(addprefix testp-,$(TARGETS))

test-hello: hello
	$(call run, ./$< +p4 10)

testp-hello: hello
	$(call run, ./$< +p$(P) $$(( ($(P) * 5) / 2 )) )
