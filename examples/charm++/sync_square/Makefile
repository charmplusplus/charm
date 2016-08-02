-include ../../common.mk
CHARMC = ../../../bin/charmc
BINARY = sync_square

CHARMCFLAGS = $(OPTS)
CHARMCLINKFLAGS = -language charm++
TESTFLAGS =  +p1

%.o: %.C

all: $(BINARY)
$(BINARY): $(patsubst %.C,%.o,$(wildcard *.C))
	$(CHARMC) $(CHARMCLINKFLAGS) -o $@ $+ 

.SECONDARY: $(patsubst %.C,%.decl.h,$(wildcard *.C))
.SECONDARY: $(patsubst %.C,%.def.h,$(wildcard *.C))

%.o: %.C %.decl.h %.def.h
	$(CHARMC) $(CHARMCFLAGS) $<

%.decl.h %.def.h: %.ci
	$(CHARMC) $(CHARMCFLAGS) $<

test: $(BINARY)
	$(call run, ./$(BINARY) $(TESTFLAGS))

clean:
	rm -f *.o *.decl.h *.def.h charmrun $(BINARY)
