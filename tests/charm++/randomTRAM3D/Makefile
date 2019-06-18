-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC = ../../../bin/charmc $(OPTS)

OBJS = tram3d.o

all: tram3d

tram3d: tram3d.o
	$(CHARMC) $(CHARMCFLAGS) -language charm++ -o tram3d tram3d.o -module NDMeshStreamer

tram3d.def.h: tram3d.ci
	$(CHARMC) $(CHARMCFLAGS) tram3d.ci

tram3d.o: tram3d.C tram3d.def.h
	$(CHARMC) $(CHARMCFLAGS) -c tram3d.C

test: tram3d
	$(call run, ./tram3d +p4 )

clean:
	rm -f *.o *.decl.h *.def.h tram3d charmrun*
