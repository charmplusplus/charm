# modify the path of CHARMC
CHARMC=../../../bin/charmc $(OPTS)

all: array

array: pgm.o
	$(CHARMC) pgm.o  -o array -language charm++

pgm.o : pgm.C pgm.def.h
	$(CHARMC) -c pgm.C

pgm.decl.h pgm.def.h: pgm.ci
	$(CHARMC)  pgm.ci

clean:
	rm -f array pgm_prof *.o conv-host *.decl.h *.def.h *~ charmrun *.log *.sts core charmrun.exe array.exe array.pdb array.ilk
