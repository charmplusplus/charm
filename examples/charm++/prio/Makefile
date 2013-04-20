# modify the path of CHARMC
CHARMC ?= ../../../bin/charmc $(OPTS)

all: pgm

pgm: pgm.o
	$(CHARMC) pgm.o  -o pgm -language charm++

pgm.o : pgm.C pgm.def.h
	$(CHARMC) -c pgm.C

pgm.decl.h pgm.def.h: pgm.ci.stamp

pgm.ci.stamp: pgm.ci
	$(CHARMC) $<
	touch $@

clean:
	rm -f pgm pgm_prof *.o conv-host *.decl.h *.def.h *.stamp *~ charmrun *.log *.sts core charmrun.exe pgm.exe pgm.pdb pgm.ilk
