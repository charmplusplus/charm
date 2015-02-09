-include ../../../common.mk
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
	$(CHARMC) $(OPTS) caseTest.ci

caseTest.o: caseTest.C caseTest.decl.h
	$(CHARMC) $(OPTS) -c $<

caseTest: caseTest.o
	$(CHARMC) -language charm++ $(OPTS) $^ -o $@

test: caseTest
	$(call run, +p$(PROCS) ./$< )
