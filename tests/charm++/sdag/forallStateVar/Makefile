-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)
ALL=forallStateVar
RUN=./charmrun

ifndef $(PROCS)
        PROCS=2
endif

.PHONY: all clean

all: $(ALL)

clean:
	-rm -rf $(wildcard *.o *~ core *.decl.h *.def.h *.txt charmrun $(ALL))

forallStateVar.decl.h: forallStateVar.ci
	$(CHARMC) $(OPTS) forallStateVar.ci

forallStateVar.o: forallStateVar.C forallStateVar.decl.h
	$(CHARMC) $(OPTS) -c $<

forallStateVar: forallStateVar.o
	$(CHARMC) -language charm++ $(OPTS) $^ -o $@

test: $(ALL)
	$(call run, +p$(PROCS) ./$< )
