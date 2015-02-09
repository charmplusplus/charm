-include ../../common.mk
# modify the path of CHARMC
CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm: pgm.o
	$(CHARMC) pgm.o  -o pgm -language charm++

test: pgm
	$(call run, +p4 ./pgm 1000000 100 )

bgtest: pgm
	$(call run, +p4 ./pgm 1000000 100 +x2 +y2 +z2 )


# compile program with trace projections - performance trace library
pgm_prof: pgm.o
	$(CHARMC) pgm.o  -o pgm_prof -language charm++ -tracemode projections

pgm.o : pgm.C pgm.h pgm.def.h
	$(CHARMC) -c pgm.C

pgm.decl.h pgm.def.h: pgm.ci
	$(CHARMC)  pgm.ci

clean:
	rm -f pgm pgm_prof *.o conv-host *.decl.h *.def.h *~ charmrun *.log *.sts core charmrun.exe pgm.exe pgm.pdb pgm.ilk
