-include ../../../common.mk
-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

OBJS=caseTest.o caseTestf.o

ALL=caseTest

ifndef $(PROCS)
	PROCS=2
endif

.PHONY: all clean

all: $(ALL)

clean:
	-rm -rf *.o *~ core *.decl.h *.def.h *.txt charmrun $(ALL)

caseTest.decl.h: caseTest.ci
	$(CHARMC) $(OPTS) -language f90charm caseTest.ci

caseTest.o: caseTest.C caseTest.decl.h
	$(CHARMC) $(OPTS) -c $<

caseTestf.o: caseTestf.f90
	$(CHARMC) -c caseTestf.f90

caseTest: $(OBJS)
	$(CHARMC) -language f90charm $(OPTS) $^ -o $@

test: caseTest
	$(call run, +p$(PROCS) ./$< )
