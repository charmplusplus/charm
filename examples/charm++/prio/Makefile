# modify the path of CHARMC
CHARMC ?= ../../../bin/charmc $(OPTS)

all: prio

prio: pgm.o
	$(CHARMC) pgm.o  -o prio -language charm++

pgm.o : pgm.C pgm.def.h
	$(CHARMC) -c pgm.C

pgm.decl.h pgm.def.h: pgm.ci.stamp

pgm.ci.stamp: pgm.ci
	$(CHARMC) $<
	touch $@

clean:
	rm -f prio pgm_prof *.o conv-host *.decl.h *.def.h *.stamp *~ charmrun *.log *.sts core charmrun.exe prio.exe prio.pdb prio.ilk
