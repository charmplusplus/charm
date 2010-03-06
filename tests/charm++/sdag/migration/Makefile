CHARMC=../../../../bin/charmc $(OPTS)
DEFAULT=test1
ALL=test1
RUN=./charmrun

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
	$(RUN) +p$(PROCS) ./test1   $(TESTOPTS)
