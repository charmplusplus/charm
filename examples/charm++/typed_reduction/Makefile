CHARMC = ../../../bin/charmc
BINARY = typed_reduction

CHARMCFLAGS =
CHARMCLINKFLAGS = -language charm++
TESTFLAGS = 

%.o: %.cc

all: $(BINARY)
$(BINARY): $(patsubst %.cc,%.o,$(wildcard *.cc))
	$(CHARMC) $(CHARMCLINKFLAGS) -o $@ $+

.SECONDARY: $(patsubst %.cc,%.decl.h,$(wildcard *.cc))
.SECONDARY: $(patsubst %.cc,%.def.h,$(wildcard *.cc))

%.o: %.cc %.decl.h %.def.h
	$(CHARMC) $(CHARMCFLAGS) $<

%.decl.h %.def.h: %.ci
	$(CHARMC) $(CHARMCFLAGS) $<

test: $(BINARY)
	./charmrun ./$(BINARY) $(TESTFLAGS)

clean:
	rm -f *.o *.decl.h *.def.h charmrun $(BINARY)
