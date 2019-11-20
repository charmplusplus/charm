-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)
OPTS ?= -optimize -production
CHARMCFLAGS = $(OPTS)

all: overhead_test

overhead_test: overhead_test.o
	$(CHARMC) $(CHARMCFLAGS) -language charm++ -o overhead_test overhead_test.o

overhead_test.o: overhead_test.C overhead_test.decl.h overhead_test.def.h
	$(CHARMC) $(CHARMCFLAGS) overhead_test.C

%.def.h %.decl.h: %.ci
	$(CHARMC) $(CHARMCFLAGS) $<

test: all
	$(call run, +p2 ./overhead_test )

clean:
	rm -f overhead_test.decl.h overhead_test.def.h overhead_test.o
	rm -f core *~
	rm -f overhead_test
	rm -f charmrun
