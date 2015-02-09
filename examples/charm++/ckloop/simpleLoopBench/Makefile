-include ../../../common.mk
#NOTE: to compile the example, the CkLoop.decl/def.h should exist
USEROPTS=-O3 -g -lpthread -fopenmp
CHARMDIR=../../../..
CHARMINC=$(CHARMDIR)/include
OPTS=-I$(CHARMINC) $(USEROPTS)
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS)
CHARMLIB=$(CHARMDIR)/lib


all: hello

hello: hello.o
	$(CHARMC) -language charm++ -o hello hello.o  -module CkLoop
	$(CHARMC) -language charm++ -o hello.prj hello.o  -tracemode projections -module CkLoop

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

hello.o: hello.C hello.decl.h 
	$(CHARMC) -c hello.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello hello.prj charmrun *.log *.sum *.sts

test: all
	$(call run, ./hello +p4 10)

bgtest: all
	$(call run, ./hello +p4 10 +x2 +y2 +z2 +cth1 +wth1)

module: $(CHARMLIB)/libmoduleCkLoop.a

$(CHARMLIB)/libmoduleCkLoop.a: CkLoop.o
	$(CHARMC)  -o $(CHARMLIB)/libmoduleCkLoop.a CkLoop.o


CkLoop.decl.h: CkLoop.ci
	$(CHARMC)  CkLoop.ci


CkLoop.o: CkLoop.C CkLoop.decl.h
	$(CHARMC) -c CkLoop.C

