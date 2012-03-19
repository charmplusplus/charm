CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm: pgm.o
	$(CHARMC) pgm.o  -o pgm -language charm++

pgm.o : pgm.C pgm.h pgm.decl.h
	$(CHARMC) -c pgm.C

pgm.decl.h: pgm.ci
	$(CHARMC) -c pgm.ci

clean:
	rm -f pgm *.o conv-host *.decl.h *.def.h *~ charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk

test: pgm
	./charmrun ./pgm +p2 1000 10 $(TESTOPTS)

bgtest: pgm
	./charmrun ./pgm +p2 1000 10 +x2 +y2 +z1 $(TESTOPTS)

