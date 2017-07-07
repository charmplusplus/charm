CHARMRUN=./charmrun
-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)
ALL=caseTest

ifndef $(PROCS)
	PROCS=2
endif

.PHONY: all clean

all: $(ALL)

clean:
	-rm -rf *.o *~ core *.decl.h *.def.h *.txt charmrun $(ALL)

caseTest.decl.h: caseTest.ci
	$(CHARMC) caseTest.ci

caseTest.o: caseTest.C caseTest.decl.h
	$(CHARMC) -c $<

caseTest: caseTest.o
	$(CHARMC) -language charm++ $^ -o $@

test-local: caseTest
	$(call run, ./$< +p$(PROCS) )
