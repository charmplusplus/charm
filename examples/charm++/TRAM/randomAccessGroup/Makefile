-include ../../../common.mk
CHARMC = ../../../../bin/charmc
BINARY = random_access

SRCFILES = $(wildcard *.C)
OBJFILES = $(SRCFILES:.C=.o)
CIFILES = $(wildcard *.ci)
HFILES = $(CIFILES:.ci=.decl.h) $(CIFILES:.ci=.def.h)
CHARMCFLAGS= $(OPTS)

all: $(BINARY)

random_access: $(OBJFILES)
	$(CHARMC) $(CHARMCFLAGS) -language charm++ -o $@ $+ -module NDMeshStreamer

.SECONDARY: $(patsubst %.C,%.decl.h,$(wildcard *.C))
.SECONDARY: $(patsubst %.C,%.def.h,$(wildcard *.C))

%.def.h %.decl.h: %.ci
	$(CHARMC) $(CHARMCFLAGS) $<

%.o: %.C

%.o: %.C $(HFILES)
	$(CHARMC) $(CHARMCFLAGS) $<

test: $(BINARY)
	$(call run, +p4 ./random_access 14 )

clean:
	rm -f *.o *.decl.h *.def.h $(BINARY) charmrun*
