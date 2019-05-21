-include ../../../common.mk
CHARMDIR=../../../..
CHARMINC=$(CHARMDIR)/include
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS)
CHARMLIB=$(CHARMDIR)/lib


all: hello

hello: hello.o
	$(CHARMC) -language charm++ -o hello hello.o  -module CkLoop -module OmpCharm

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

hello.o: hello.C hello.decl.h 
	$(CHARMC) -c hello.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello hello.prj charmrun *.log *.sum *.sts

test: all
	$(call run, ./hello +p4 -a 8)
