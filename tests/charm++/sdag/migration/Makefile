-include ../../../common.mk
-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)
DEFAULT=test1
ALL=test1

ifndef $(PROCS)
	PROCS=2
endif

.PHONY: all clean clobber run seq test

all: $(ALL)

clean:
	-rm -rf *.o *~ core *.decl.h *.def.h *.txt charmrun $(ALL) test1.exe test1.pdb test1.ilk

test1.decl.h: test1.ci
	$(CHARMC) $(OPTS) test1.ci

test1.o: test1.C test1.decl.h
	$(CHARMC) $(OPTS) -c $<

test1: test1.o
	$(CHARMC) -language charm++ $(OPTS) $^ -o $@

test: test1
	$(call run, +p$(PROCS) ./test1 )
