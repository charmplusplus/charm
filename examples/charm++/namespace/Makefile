CHARMRUN=./charmrun
-include ../../common.mk
CHARMC = ../../../bin/charmc
BINARY = inherit

CHARMCFLAGS = $(OPTS)
CHARMCLINKFLAGS = -language charm++ $(OPTS)
TESTFLAGS =  +p1

%.o: %.C

all: $(BINARY)
$(BINARY): $(patsubst %.C,%.o,$(wildcard *.C))
	$(CHARMC) $(CHARMCLINKFLAGS) -o $@ $+

%.o: %.C %.decl.h %.def.h
	$(CHARMC) $(CHARMCFLAGS) $<

Inherit.o: C.decl.h C.def.h B.decl.h B.def.h A.decl.h A.def.h
C.o: B.decl.h B.def.h A.decl.h A.def.h
B.o: A.decl.h A.def.h

%.decl.h %.def.h: %.ci
	$(CHARMC) $(CHARMCFLAGS) $<

test-local: $(BINARY)
	$(call run, ./$(BINARY) $(TESTFLAGS))

clean:
	rm -f *.o *.decl.h *.def.h charmrun $(BINARY)
