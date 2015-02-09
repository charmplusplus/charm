-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)
ALL=sdagCase

ifndef $(PROCS)
        PROCS=2
endif

.PHONY: all clean

all: $(ALL)

clean:
	-rm -rf $(wildcard *.o *~ core *.decl.h *.def.h *.txt charmrun $(ALL))

sdagCase.decl.h: sdagCase.ci
	$(CHARMC) $(OPTS) sdagCase.ci

sdagCase.o: sdagCase.C sdagCase.decl.h
	$(CHARMC) $(OPTS) -c $<

sdagCase: sdagCase.o
	$(CHARMC) -language charm++ $(OPTS) $^ -o $@

test: $(ALL)
	$(call run, +p$(PROCS) ./$< )
