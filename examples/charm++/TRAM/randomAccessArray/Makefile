-include ../../../common.mk
CHARMC = ../../../../bin/charmc
BINARY = random_access

SRCFILES = $(wildcard *.cc)
OBJFILES = $(SRCFILES:.cc=.o)
CIFILES = $(wildcard *.ci)
HFILES = $(CIFILES:.ci=.decl.h) $(CIFILES:.ci=.def.h)
OPTS ?= -optimize -production
CHARMCFLAGS= $(OPTS)

all: $(BINARY)

random_access: $(OBJFILES)
	$(CHARMC) $(CHARMCFLAGS) -language charm++ -o $@ $+ -module NDMeshStreamer

.SECONDARY: $(patsubst %.cc,%.decl.h,$(wildcard *.cc))
.SECONDARY: $(patsubst %.cc,%.def.h,$(wildcard *.cc))

%.def.h %.decl.h: %.ci
	$(CHARMC) $(CHARMCFLAGS) $<

%.o: %.cc

%.o: %.cc $(HFILES)
	$(CHARMC) $(CHARMCFLAGS) $<

test: $(BINARY)
	$(call run, +p4 ./random_access 14 8 )

clean:
	rm -f *.o *.decl.h *.def.h $(BINARY) charmrun*
