# modify the path of CHARMC
CHARMC=../../../../bin/charmc $(OPTS)

pgm: pgm.o
	$(CHARMC) pgm.o  -o pgm -language charm++

# compile program with trace projections - performance trace library
pgm_prof: pgm.o
	$(CHARMC) pgm.o  -o pgm_prof -language charm++ -tracemode projections

pgm.o : pgm.C pgm.h pgm.def.h
	$(CHARMC) -c pgm.C

pgm.decl.h pgm.def.h: pgm.ci
	$(CHARMC)  pgm.ci

clean:
	rm -f pgm pgm_prof *.o conv-host *.decl.h *.def.h *~ charmrun *.log *.sts core
