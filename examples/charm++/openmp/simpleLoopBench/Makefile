-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
CHARMDIR=../../../..
CHARMINC=$(CHARMDIR)/include
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS)
CHARMLIB=$(CHARMDIR)/lib

TARGETS :=
ifeq (1,$(CMK_BUILD_OMP))
  TARGETS += hello
endif

all: $(TARGETS)

hello: hello.o
	$(CHARMC) -language charm++ -o hello hello.o  -module CkLoop -module OmpCharm

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

hello.o: hello.C hello.decl.h 
	$(CHARMC) -c hello.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello hello.prj charmrun *.log *.sum *.sts

test: $(addprefix test-,$(TARGETS))

testp: $(addprefix testp-,$(TARGETS))

test-hello: hello
	$(call run, ./$< +p4 -a 8)

testp-hello: hello
	$(call run, ./$< +p$(P) -a $$(( $(P) * 2 )) )
