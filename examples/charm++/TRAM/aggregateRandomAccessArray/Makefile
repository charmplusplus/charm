-include ../../../common.mk
CHARMC = ../../../../bin/charmc
BINARY = random_access

CHARMCFLAGS = $(OPTS)

all: $(BINARY)

random_access: randomAccess.o
	$(CHARMC) $(CHARMCFLAGS) -language charm++ -o $@ $+ -module NDMeshStreamer

.SECONDARY: $(patsubst %.C,%.decl.h,$(wildcard *.C))
.SECONDARY: $(patsubst %.C,%.def.h,$(wildcard *.C))

randomAccess.def.h randomAccess.decl.h: randomAccess.ci.stamp

randomAccess.ci.stamp: randomAccess.ci
	$(CHARMC) $(CHARMCFLAGS) $<
	touch $@

randomAccess.o: randomAccess.C randomAccess.decl.h randomAccess.def.h
	$(CHARMC) $(CHARMCFLAGS) -c randomAccess.C

test: $(BINARY)
	$(call run, +p4 ./random_access 14 8 )

clean:
	rm -f *.o *.decl.h *.def.h $(BINARY) charmrun* *.stamp
